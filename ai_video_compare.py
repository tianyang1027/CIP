import cv2
import numpy as np
import base64
import json
import os
from typing import List, Dict
from openai import OpenAI  # ensure this matches your installed SDK

# init client (relies on OPENAI_API_KEY env var)
client = OpenAI()

def extract_key_frames(video_path: str, frame_skip: int = 5, diff_thresh_small=0.01, diff_thresh_large=0.25) -> List[Dict]:
    """
    Extract key frames (time + image bytes) from video using frame-difference.
    Returns a list of dicts: {"timestamp": float, "image_b64": "<base64 png>"}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    prev_gray = None
    frame_index = 0
    raw_events = []  # store (frame_index, change_ratio)

    # pass 1: detect frames with notable change
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        change_ratio = float(np.sum(thresh > 0) / thresh.size)

        if change_ratio > diff_thresh_small:
            raw_events.append((frame_index, change_ratio))
        prev_gray = gray

    # choose representative frames for events (filter events too close)
    selected_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_timestamp = -999.0
    for (frame_idx, change_ratio) in raw_events:
        timestamp = frame_idx / fps
        # keep events at least 0.5s apart
        if timestamp - prev_timestamp < 0.5:
            continue
        # seek frame and read exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # encode to PNG base64
        _, buffer = cv2.imencode(".png", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        selected_frames.append({
            "timestamp": round(timestamp, 2),
            "change_ratio": round(change_ratio, 3),
            "image_b64": img_b64
        })
        prev_timestamp = timestamp

    cap.release()
    return selected_frames

def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_model_prompt(standard_steps: List[Dict], key_frames: List[Dict], max_frames: int = 6) -> List[Dict]:
    """
    Build messages for the vision+language model.
    - include standard_steps (JSON)
    - include a small set of key frames (data urls with timestamps)
    Ask the model to:
      1) reconstruct user's action sequence from frames,
      2) compare to standard_steps,
      3) return a single-line result: either "Correct" or "Wrong: Step X [Title] - <reason>".
    """
    std_steps_str = json.dumps(standard_steps, ensure_ascii=False)
    # limit number of frames to avoid too large prompt
    frames_to_send = key_frames[:max_frames]

    # build the user content with instructions and the frames as data URLs
    content_parts = [
        "You are an expert web interaction evaluator. You will be given:",
        "1) the standard workflow as JSON (array of steps, each step has 'step', 'title', and 'expected_actions' which is an array of action tokens).",
        "2) a small set of key video frames extracted from a user's screen recording, with timestamps.",
        "",
        "Task:",
        " - From the frames, infer the user's action sequence (use your best judgement: clicks, input events, navigations).",
        " - Compare the inferred sequence to the standard workflow.",
        " - If ALL steps match (including expected actions and order), return exactly: Correct",
        " - Otherwise (stop at the first incorrect step), return exactly one line in this format:",
        '   Wrong: Step <N> [<Title>] - <Natural language reason describing what the user did wrong (concise)>',
        "",
        "Do not output anything else. The reason should be a natural English sentence, not JSON.",
        "",
        "Standard workflow JSON:",
        std_steps_str,
        "",
        "Key frames (timestamp seconds and image data URLs):"
    ]

    for idx, f in enumerate(frames_to_send, start=1):
        data_url = f"data:image/png;base64,{f['image_b64']}"
        content_parts.append(f"Frame {idx} @ {f['timestamp']}s (change_ratio={f['change_ratio']}): {data_url}")

    content = "\n".join(content_parts)

    messages = [
        {"role": "system", "content": "You are a careful, precise evaluator of user web interactions."},
        {"role": "user", "content": content}
    ]
    return messages

def call_model_and_get_result(messages: List[Dict], model: str = "gpt-5-vision", timeout: int = 120) -> str:
    """
    Call the model. Expect a single-line response exactly either "Correct" or "Wrong: Step X ..."
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages
    )
    # extract text content (SDK may vary — adapt if needed)
    body = resp.choices[0].message.content
    # ensure single-line trimmed
    result = body.strip().splitlines()[0].strip()
    return result

def evaluate_video_against_standard(video_path: str, standard_file: str) -> str:
    standard_steps = load_json(standard_file)
    key_frames = extract_key_frames(video_path)
    if not key_frames:
        # if no key frames detected, we still want AI to consider the whole video; but here we return an explanatory failure
        return "Wrong: Step 0 [No detectable events] - The video does not contain detectable UI changes; unable to infer user actions."

    messages = prepare_model_prompt(standard_steps, key_frames)
    result = call_model_and_get_result(messages)
    return result

if __name__ == "__main__":
    # paths
    VIDEO_PATH = "recording.mp4"          # replace with your video path
    STANDARD_JSON = "standard_steps.json" # replace with your standard steps json

    final_result = evaluate_video_against_standard(VIDEO_PATH, STANDARD_JSON)
    print(final_result)
