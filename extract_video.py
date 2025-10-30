import cv2
import numpy as np
import json
import os

def extract_actions_from_video(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_frame = None
    actions = []
    frame_idx = 0
    action_id = 0

    print(f"Analyzing {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # 帧差法检测变化（点击/跳转）
        diff = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
        change_ratio = np.sum(thresh > 0) / thresh.size

        # 如果页面有大范围变化 => 页面跳转
        if change_ratio > 0.25:
            actions.append({
                "timestamp": round(frame_idx / fps, 2),
                "action": "navigate_or_refresh",
                "confidence": round(change_ratio, 3)
            })
        # 如果变化局部较小 => 点击事件
        elif 0.01 < change_ratio < 0.25:
            actions.append({
                "timestamp": round(frame_idx / fps, 2),
                "action": "click_event",
                "confidence": round(change_ratio, 3)
            })

        prev_frame = gray

    cap.release()

    # 去重 + 简化动作
    filtered_actions = []
    for i, act in enumerate(actions):
        if i == 0 or act["timestamp"] - actions[i - 1]["timestamp"] > 0.5:
            filtered_actions.append(act)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_actions, f, indent=2, ensure_ascii=False)

    print(f"✅ 动作提取完成，共检测到 {len(filtered_actions)} 个事件，结果保存在 {output_json}")
    return filtered_actions