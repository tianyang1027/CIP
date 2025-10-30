import pytesseract
from PIL import Image
import cv2

# OCR 识别文字
def ocr_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img)
    return text.strip()
