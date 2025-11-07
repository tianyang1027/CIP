import cv2
import numpy as np
from difflib import SequenceMatcher
from PIL import Image
import pytesseract

def text_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def template_match(full_image_path, template_path, threshold=0.8):
    img = cv2.imread(full_image_path, 0)
    template = cv2.imread(template_path, 0)
    if img is None or template is None:
        return False
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return len(loc[0]) > 0

def ocr_text_match(screenshot_path, expected_text):
    try:
        text = pytesseract.image_to_string(Image.open(screenshot_path), lang='chi_sim')
        return expected_text in text
    except:
        return False

def compare_steps_ai_reason(standard_steps, actual_steps, text_threshold=0.8, image_threshold=0.8):
    for i, (std_step, act_step) in enumerate(zip(standard_steps, actual_steps), start=1):
        # 文本判断
        sim = text_similarity(std_step['text'], act_step['text'])
        if sim < text_threshold:
            reason = f"步骤 {i}: 标准要求 '{std_step['text']}'，实际描述为 '{act_step['text']}'。可能操作未完成或顺序有误。"
            return {"result": False, "step": i, "reason": reason}

        # 模板判断
        if std_step.get('image_path') and act_step.get('image_path'):
            if not template_match(act_step['image_path'], std_step['image_path'], image_threshold):
                reason = f"步骤 {i}: 操作区域应显示关键元素（如菜单或按钮），但截图未显示该区域。可能操作未执行。"
                return {"result": False, "step": i, "reason": reason}

        # OCR 判断
        if std_step.get('ocr_text') and act_step.get('image_path'):
            if not ocr_text_match(act_step['image_path'], std_step['ocr_text']):
                reason = f"步骤 {i}: 操作要求界面包含文字 '{std_step['ocr_text']}'，但截图未显示该文字，说明操作未完成。"
                return {"result": False, "step": i, "reason": reason}

    return {"result": True, "reason": "所有操作步骤符合标准。"}

# ------------------------------
# 示例
# ------------------------------
standard_steps = [
    {
        "text": "右键新建文件夹",
        "image_path": "templates/right_click_menu.png",
        "ocr_text": "新建文件夹"
    },
    {
        "text": "打开浏览器并进入网址 http://example.com",
        "image_path": None,
        "ocr_text": "http://example.com"
    }
]

actual_steps = [
    {
        "text": "右键菜单没有新建文件夹",
        "image_path": "screenshots/user_right_click.png"
    },
    {
        "text": "打开浏览器",
        "image_path": "screenshots/user_browser.png"
    }
]

result = compare_steps_ai_reason(standard_steps, actual_steps)
print(result)
