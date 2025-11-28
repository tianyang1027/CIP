import cv2
import numpy as np
import requests

def split_video_to_frames(video_path):
    """
    Splits a video into frames and stores each frame as an image in memory (as NumPy arrays).
    
    :param video_path: Path to the video file.
    :return: A list containing the video frames (each frame is a NumPy array).
    """
    frames = []

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Append each frame to the frames list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    print(f"Video split completed, {len(frames)} frames generated.")
    return frames

def check_video_url(url):
    """
    Checks if a video URL can be opened and is not a dead link.

    :param url: The URL of the video.
    :return: True if the video URL can be opened, False otherwise.
    """
    # Step 1: Check if the URL is reachable (HTTP status code check)
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            print(f"Error: URL returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error: Unable to reach the URL. Exception: {e}")
        return False

    # Step 2: Try to open the video with OpenCV
    cap = cv2.VideoCapture(url)
    
    # Check if the video capture object was successfully opened
    if not cap.isOpened():
        print("Error: Failed to open video from URL.")
        return False
    
    # Successfully opened the video
    cap.release()
    return True