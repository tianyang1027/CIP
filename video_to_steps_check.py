import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import pytesseract
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple

# -----------------------------
# 配置（可改）
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FRAME_DIFF_THRESHOLD = 30          # 帧差阈值，控制关键帧检测灵敏度
MIN_FRAME_INTERVAL = 0.5           # 秒，最小帧间隔（防止过密采样）
SIMILARITY_THRESHOLD = 0.28       # CLIP 图像相似度判定阈值（需经验调节）
OCR_KEYWORD_MATCH_RATIO = 0.5     # OCR 命中关键字比例阈值

# -----------------------------
# 工具函数
# -----------------------------
def load_standard_steps(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        steps = json.load(f)
    return steps

def extract_keyframes(video_path: str, out_dir: str = None,
                      diff_threshold=FRAME_DIFF_THRESHOLD,
                      min_interval=MIN_FRAME_INTERVAL) -> List[Tuple[int, Image.Image]]:
    """
    从视频中提取关键帧（简单基于帧差 + 时间间隔）
    返回 list of (frame_index, PIL.Image)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = int(max(1, fps * min_interval))
    keyframes = []
    prev_gray = None
    frame_idx = 0
    saved_idx = 0

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Extract frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0 and prev_gray is not None:
            frame_idx += 1
            pbar.update(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            take = True
        else:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = np.mean(diff)
            take = mean_diff > diff_threshold

        if take:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keyframes.append((frame_idx, pil))
            saved_idx += 1
            prev_gray = gray

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    # 可选：保存到 out_dir
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, (fi, img) in enumerate(keyframes):
        if out_dir:
            img.save(os.path.join(out_dir, f"frame_{i:03d}_{fi}.png"))
    return keyframes

# -----------------------------
# CLIP embedding
# -----------------------------
class CLIPEmbedder:
    def __init__(self, model_name=CLIP_MODEL_NAME, device=DEVICE):
        print("Loading CLIP model:", model_name, "device:", device)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu()

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """
    A: (n, d), B: (m, d)
    return (n, m) matrix of cosine similarities
    """
    A = A.numpy()
    B = B.numpy()
    sim = np.dot(A, B.T)
    return sim

# -----------------------------
# OCR & keyword check
# -----------------------------
def ocr_extract_text(pil_image: Image.Image) -> str:
    # 可根据需要传入语言参数，如 lang='chi_sim+eng'
    text = pytesseract.image_to_string(pil_image)
    return text

def description_keywords(desc: str) -> List[str]:
    # 简单分词：以空格/中文标点/英文分隔（可替换为更强的中文分词）
    import re
    tokens = re.split(r"[ \t,，。.!？?；;:\n]+", desc)
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 2]  # 过滤太短的词
    return tokens

def ocr_matches_description(ocr_text: str, description: str) -> Tuple[bool, float, List[str]]:
    ocr_text_lower = ocr_text.lower()
    kws = description_keywords(description)
    if not kws:
        return False, 0.0, []
    matched = [k for k in kws if k.lower() in ocr_text_lower]
    ratio = len(matched) / len(kws)
    return ratio >= OCR_KEYWORD_MATCH_RATIO, ratio, matched

# -----------------------------
# 匹配逻辑
# -----------------------------
def match_steps_to_frames(steps: List[Dict], frames: List[Tuple[int, Image.Image]], embedder: CLIPEmbedder):
    # 1. 计算标准图片的 embedding 与文本embedding
    step_imgs = []
    step_texts = []
    for s in steps:
        img_path = s.get("image")
        if img_path and os.path.exists(img_path):
            step_imgs.append(Image.open(img_path).convert("RGB"))
        else:
            # 占位白图（如果没有图片，就用空白）
            step_imgs.append(Image.new("RGB", (640, 480), color=(255,255,255)))
        step_texts.append(s.get("description", ""))

    frame_images = [f[1] for f in frames]

    step_img_embs = embedder.encode_images(step_imgs)         # (N_steps, d)
    step_text_embs = embedder.encode_texts(step_texts)        # (N_steps, d)
    frame_img_embs = embedder.encode_images(frame_images)     # (N_frames, d)

    # 图像相似度矩阵： steps x frames
    sim_img = cosine_similarity_matrix(step_img_embs, frame_img_embs)
    # 文本-帧相似度也可以用（文本embedding 与帧embedding）
    sim_txt = cosine_similarity_matrix(step_text_embs, frame_img_embs)

    results = []
    last_matched_frame_idx = -1
    for i, s in enumerate(steps):
        # 先找在 last_matched_frame_idx 之后最高的匹配帧
        sims = sim_img[i] * 0.7 + sim_txt[i] * 0.3   # 权重可调：图像优先，文本次之
        # 只考虑序号大于 last_matched_frame_idx 的帧（保持顺序）
        candidate_indices = [j for j in range(len(frames)) if frames[j][0] > last_matched_frame_idx]
        if not candidate_indices:
            # 没有剩余帧
            results.append({
                "step": s.get("step"),
                "expected_description": s.get("description"),
                "matched": False,
                "reason": "没有剩余视频帧可匹配，可能后续步骤缺失或视频截断"
            })
            continue
        cand_sims = [(j, sims[j]) for j in candidate_indices]
        # 找出最大相似度
        best_j, best_score = max(cand_sims, key=lambda x: x[1])
        best_frame_idx, best_frame_img = frames[best_j]
        reason_msgs = []

        # 1) 用相似度判断是否命中
        if best_score >= SIMILARITY_THRESHOLD:
            matched = True
        else:
            matched = False
            reason_msgs.append(f"CLIP 相似度低 ({best_score:.3f} < {SIMILARITY_THRESHOLD})")

        # 2) OCR 检查：抽取 frame 文本，看是否包含描述关键词
        ocr_text = ocr_extract_text(best_frame_img)
        ocr_ok, ocr_ratio, matched_kws = ocr_matches_description(ocr_text, s.get("description", ""))
        if not ocr_ok:
            reason_msgs.append(f"OCR 未匹配到足够关键词 (命中 {ocr_ratio*100:.0f}%，样例关键词: {matched_kws})")

        # 3) SSIM 或视觉相似度补充（可选）
        # 转成灰度并计算 SSIM（与标准图）
        try:
            std_img = Image.open(s.get("image")).convert("L")
            frame_gray = best_frame_img.convert("L")
            # 调整大小一致
            std_arr = np.array(std_img.resize((320, 240)))
            frame_arr = np.array(frame_gray.resize((320, 240)))
            ssim_val = ssim(std_arr, frame_arr)
        except Exception as e:
            ssim_val = None

        # 综合判定：要求 matched by CLIP 或 OCR 命中（两者之一）且顺序正确
        final_ok = (matched or ocr_ok)
        if final_ok:
            reason = "匹配成功"
            last_matched_frame_idx = best_frame_idx
        else:
            reason = "; ".join(reason_msgs) if reason_msgs else "未匹配到相应步骤画面"

        results.append({
            "step": s.get("step"),
            "expected_description": s.get("description"),
            "matched_frame_index": int(best_frame_idx),
            "clip_score": float(best_score),
            "ssim": float(ssim_val) if ssim_val is not None else None,
            "ocr_ratio": float(ocr_ratio),
            "ocr_matched_keywords": matched_kws,
            "matched": bool(final_ok),
            "reason": reason
        })

    return results

# -----------------------------
# 主流程
# -----------------------------
def run_check(standard_json: str, video_path: str, temp_frames_dir: str = "extracted_frames"):
    steps = load_standard_steps(standard_json)
    frames = extract_keyframes(video_path, out_dir=temp_frames_dir)

    if len(frames) == 0:
        print("未检测到关键帧，请检查视频或调整 diff_threshold / min_interval")
        return []

    embedder = CLIPEmbedder()
    results = match_steps_to_frames(steps, frames, embedder)
    return results

# -----------------------------
# CLI 用法
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", required=True, help="标准步骤 JSON 文件路径")
    parser.add_argument("--video", required=True, help="实际操作视频路径")
    parser.add_argument("--out", default="report.json", help="输出报告文件（JSON）")
    args = parser.parse_args()

    report = run_check(args.standard, args.video, temp_frames_dir="frames_out")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入", args.out)
