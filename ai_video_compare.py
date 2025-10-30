import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from paddleocr import PaddleOCR
from transformers import CLIPProcessor, CLIPModel
import torch
import json


# ===== 参数配置 =====
FRAME_DIFF_THRESHOLD = 12       # 图像像素变化阈值
MIN_FRAME_INTERVAL = 1.0        # 每隔几秒检测一帧
SIMILARITY_THRESHOLD = 0.25     # 匹配相似度阈值
OUTPUT_DIR = "output"


# ===== 初始化模型 =====
print("🧠 加载 CLIP 与 OCR 模型中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch')


# ===== OCR 提取文字 =====
def ocr_extract_text(pil_image: Image.Image) -> str:
    np_img = np.array(pil_image.convert("RGB"))
    result = ocr_engine.ocr(np_img, cls=True)
    texts = []
    for line in result:
        for _, (text, conf) in line:
            if conf > 0.5:
                texts.append(text)
    return " ".join(texts)


# ===== 提取关键帧（图像差异 + OCR文字差异） =====
def extract_keyframes(video_path: str, out_dir: str = None,
                      diff_threshold=FRAME_DIFF_THRESHOLD,
                      min_interval=MIN_FRAME_INTERVAL,
                      ocr_diff=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = int(max(1, fps * min_interval))
    keyframes = []
    prev_gray = None
    prev_text = ""
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="🎞️ 提取关键帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0 and prev_gray is not None:
            frame_idx += 1
            pbar.update(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.mean(cv2.absdiff(gray, prev_gray)) if prev_gray is not None else 999

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = ""
        if ocr_diff and frame_idx % int(fps * 2) == 0:
            try:
                text = ocr_extract_text(pil)
            except Exception:
                text = ""

        text_changed = (ocr_diff and len(text) > 0 and text != prev_text)
        take = (diff > diff_threshold) or text_changed or prev_gray is None

        if take:
            keyframes.append((frame_idx, pil))
            prev_gray = gray
            prev_text = text

            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                pil.save(os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg"))

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return keyframes


# ===== CLIP 匹配函数 =====
def clip_similarity(image: Image.Image, text: str) -> float:
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.softmax(dim=1).max().item()


# ===== 主逻辑：匹配操作文档步骤 =====
def check_operation(video_path: str, standard_steps: list, output_dir=OUTPUT_DIR):
    keyframes = extract_keyframes(video_path, out_dir=os.path.join(output_dir, "frames"))
    results = []

    for idx, step in enumerate(tqdm(standard_steps, desc="🧩 匹配操作步骤")):
        step_text = step.get("description", "")
        step_img = step.get("image", None)

        best_score = 0
        best_frame_idx = None
        matched = False

        for frame_idx, frame_img in keyframes:
            score = clip_similarity(frame_img, step_text)
            if score > best_score:
                best_score = score
                best_frame_idx = frame_idx

        matched = best_score >= SIMILARITY_THRESHOLD
        reason = "操作一致 ✅" if matched else "未检测到对应操作 ❌"

        results.append({
            "step": idx + 1,
            "expected_description": step_text,
            "clip_score": round(best_score, 3),
            "matched": matched,
            "reason": reason,
            "frame_index": best_frame_idx
        })

    save_html_report(results, output_dir)
    return results


# ===== HTML 报告生成 =====
def save_html_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "report.html")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>网页操作比对报告</title></head><body>")
        f.write("<h1>网页操作比对报告</h1>")
        for r in results:
            color = "green" if r["matched"] else "red"
            f.write(f"<h3 style='color:{color}'>步骤 {r['step']} - {r['reason']}</h3>")
            f.write(f"<p>描述：{r['expected_description']}<br>")
            f.write(f"相似度：{r['clip_score']}</p><hr>")
        f.write("</body></html>")

    print(f"✅ 报告生成完成：{html_path}")


# ===== 示例入口 =====
if __name__ == "__main__":
    # 这里举例：标准操作文档结构
    # 你可以把它改成从 JSON 文件读取
    standard_steps = [
        {"description": "打开浏览器并访问登录页"},
        {"description": "输入用户名密码并点击登录"},
        {"description": "在首页点击‘数据管理’选项"},
        {"description": "进入查询界面并输入查询条件"},
        {"description": "点击‘查询’按钮查看结果"}
    ]

    video_path = "your_video.mp4"  # 这里换成你的录屏文件路径
    results = check_operation(video_path, standard_steps)
    print(json.dumps(results, ensure_ascii=False, indent=2))
