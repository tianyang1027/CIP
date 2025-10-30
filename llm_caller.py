from openai import OpenAI
import base64

# Initialize OpenAI client
client = OpenAI()

# Path to the video recording
video_path = "recording.mp4"

# Read and encode video to base64
with open(video_path, "rb") as f:
    video_base64 = base64.b64encode(f.read()).decode("utf-8")

# Send video to GPT-5 Vision for action extraction
response = client.chat.completions.create(
    model="gpt-5-vision",  # or "gpt-4o"
    messages=[
        {
            "role": "system",
            "content": "You are a web operation recognition expert. Output the user's action sequence in JSON format."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please identify the steps performed in the video, including clicks, inputs, and page navigations. Output in JSON format: [{timestamp, action}]"
                },
                {
                    "type": "video",
                    "video_url": f"data:video/mp4;base64,{video_base64}"
                }
            ]
        }
    ]
)

# Print the AI-generated action sequence
print(response.choices[0].message.content)
