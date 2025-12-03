# =========================================
# 4. Ask GPT-4.1 Vision whether a frame matches a step
# =========================================
client = OpenAI(api_key="YOUR_API_KEY")

def check_frame_operation(frame, step_description):
    """
    Ask GPT-4.1 Vision whether a frame matches the given operation step.
    """
    image_b64 = frame_to_base64(frame)

    prompt = f"""
You are an operation verification assistant.
Determine whether the image is performing the following step:

Step description: "{step_description}"

Answer format:
- yes or no
- short explanation
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}"
                    }
                ]
            }
        ]
    )

    return response.choices[0].message["content"]


# =========================================
# 5. Check entire video for the operation step
# =========================================
def check_video_steps(video_path, step_description, frame_interval=30):
    """
    Check if any frame in the video satisfies a given operation step.
    frame_interval: check one frame every N frames (saves API cost)
    """

    frames = split_video_to_frames(video_path)
    print("Start checking frames...")

    for idx in range(0, len(frames), frame_interval):
        print(f"Checking frame {idx}/{len(frames)}...")
        result = check_frame_operation(frames[idx], step_description)

        print("GPT result:", result)

        if "yes" in result.lower():
            print("\n✔ Operation found in video!")
            return True

    print("\n✘ Operation NOT found in video.")
    return False
