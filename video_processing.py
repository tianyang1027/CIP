import cv2
from utils import ocr_image

def extract_video_operations(video_path, fps_sample=1):
    """
    提取视频操作事件：点击/输入/报错
    video_path: 视频路径
    fps_sample: 每秒抽帧
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    operations = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:  # 每秒抽帧
            text = ocr_image(frame)
            # 简单规则识别操作类型
            if "报错" in text or "错误" in text:
                operations.append({"frame": frame_count, "type": "error", "content": text})
            elif "点击" in text:
                operations.append({"frame": frame_count, "type": "click", "content": text})
            elif "输入" in text:
                operations.append({"frame": frame_count, "type": "input", "content": text})
        frame_count += 1

    cap.release()
    return operations
