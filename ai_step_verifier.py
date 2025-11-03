"""
智能操作步骤比对系统（Azure GPT-4o 视觉理解版）

支持：
1️⃣ 文本语义比对（SentenceTransformer）
2️⃣ 图像相似度（ResNet）
3️⃣ OCR检测（PaddleOCR）
4️⃣ LLM解释 + 视觉理解（Azure GPT-4o）
"""

import os
import base64
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import torchvision.models as models
import torchvision.transforms as transforms
from paddleocr import PaddleOCR
from openai import AzureOpenAI


# =========================================================
# 🔧 Azure OpenAI 配置（请改这里）
# =========================================================
AZURE_OPENAI_ENDPOINT = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
AZURE_OPENAI_API_KEY = "YOUR_AZURE_KEY"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"   # GPT-4o 支持图像输入
API_VERSION = "2024-05-01-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION
)


# =========================================================
# 1️⃣ 文本语义相似度
# =========================================================
text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def semantic_text_similarity(a, b):
    e1 = text_model.encode(a, convert_to_tensor=True)
    e2 = text_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()


# =========================================================
# 2️⃣ 图像特征相似度
# =========================================================
image_model = models.resnet18(pretrained=True)
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_similarity(path1, path2):
    try:
        i1 = transform(Image.open(path1).convert("RGB")).unsqueeze(0)
        i2 = transform(Image.open(path2).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            f1 = image_model(i1).flatten()
            f2 = image_model(i2).flatten()
        return torch.nn.functional.cosine_similarity(f1, f2, dim=0).item()
    except Exception:
        return 0.0


# =========================================================
# 3️⃣ OCR识别
# =========================================================
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def ocr_text_match(img_path, expected):
    try:
        result = ocr.ocr(img_path, cls=True)
        text = " ".join([r[1][0] for r in result[0]])
        return (expected in text), text
    except Exception:
        return False, ""


# =========================================================
# 4️⃣ LLM 图像+文字理解（GPT-4o）
# =========================================================
def llm_understand_image(step_text, image_path):
    """
    用 GPT-4o 判断：图片内容是否符合文字描述
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""
请阅读下面的步骤描述，并分析图片是否表现了该操作。

步骤描述：
{step_text}

请回答：
- 图片是否符合该描述（是/否）
- 简要说明理由（中文）
"""

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "你是一个视觉理解专家，擅长分析截图是否匹配操作描述。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_b64}"
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.2
        )

        reply = response.choices[0].message.content.strip()
        match = "是" in reply or "符合" in reply
        return match, reply

    except Exception as e:
        return False, f"[LLM错误: {e}]"


# =========================================================
# 5️⃣ 主逻辑
# =========================================================
def compare_steps_ai_reason(standard_steps, actual_steps, text_threshold=0.75):
    results = []

    for i, (std, act) in enumerate(zip(standard_steps, actual_steps), start=1):
        text_sim = semantic_text_similarity(std["text"], act["text"])
        img_sim = None
        ocr_match = True
        reason = ""

        # 如果标准步骤有图像
        if std.get("image_path") and act.get("image_path"):
            img_sim = image_similarity(std["image_path"], act["image_path"])

        # 如果标准步骤有OCR要求
        if std.get("ocr_text") and act.get("image_path"):
            ocr_match, ocr_text = ocr_text_match(act["image_path"], std["ocr_text"])

        # 🧠 如果标准步骤没有图像但有描述，使用 GPT-4o 判断截图是否符合
        if not std.get("image_path") and act.get("image_path"):
            llm_match, llm_reason = llm_understand_image(std["text"], act["image_path"])
            if not llm_match:
                results.append({
                    "step": i,
                    "result": False,
                    "reason": f"❌ 步骤 {i}：截图未表现出 '{std['text']}' 的操作。\nLLM分析：{llm_reason}"
                })
                continue

        # 正常文本相似度判定
        if text_sim < text_threshold or not ocr_match:
            results.append({
                "step": i,
                "result": False,
                "reason": f"❌ 步骤 {i}：文字或OCR未匹配。\n标准：{std['text']} 实际：{act['text']}"
            })
            continue

        results.append({
            "step": i,
            "result": True,
            "reason": f"✅ 步骤 {i} 匹配成功"
        })

    return results


# =========================================================
# 6️⃣ 示例
# =========================================================
if __name__ == "__main__":
    standard_steps = [
        {"text": "右键新建文件夹", "image_path": None},
        {"text": "打开浏览器并进入 http://example.com", "image_path": None}
    ]
    actual_steps = [
        {"text": "用户右键点击", "image_path": "screenshots/right_click.png"},
        {"text": "用户打开浏览器", "image_path": "screenshots/browser.png"}
    ]

    result = compare_steps_ai_reason(standard_steps, actual_steps)
    for r in result:
        print(r)
