import os
import base64
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI
import json

# Supported image formats
SUPPORTED_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
MAX_IMAGE_SIZE_MB = 20

def get_file_size_in_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def get_image_size_in_mb(data):
    return len(data) / (1024 * 1024)

def encode_image_to_base64(path):
    """Encode an image to base64. Compress/resize if larger than 20MB."""
    if not path or not os.path.exists(path):
        return None
    if get_file_size_in_mb(path) <= MAX_IMAGE_SIZE_MB:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    with Image.open(path) as img:
        img_format = "JPEG"
        img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format=img_format, optimize=True, quality=85)
        image_data = buffer.getvalue()
        while get_image_size_in_mb(image_data) > MAX_IMAGE_SIZE_MB:
            new_width = int(img.width * 0.9)
            new_height = int(img.height * 0.9)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            buffer = BytesIO()
            img.save(buffer, format=img_format, optimize=True, quality=85)
            image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode("utf-8")

def check_step(client, step_number, standard_step, actual_step):
    """Perform a single step check with text + image."""
    standard_base64 = encode_image_to_base64(standard_step.get('image_path'))
    actual_base64 = encode_image_to_base64(actual_step.get('image_path'))

    messages = [
        {"role": "system", "content": (
            "You are a computer operation step checking assistant. "
            "Compare the actual step with the standard step and determine if it matches. "
            "Return a JSON object strictly following this format:\n"
            "{\n  \"step_number\": <step number>,\n  \"result\": \"Correct\" | \"Incorrect\",\n"
            "  \"reason\": \"<If incorrect, explain the reason; if correct, can be empty or omitted>\"\n}"
        )},
        {"role": "user", "content": [
            {"type": "text", "text": f"Step number: {step_number}"},
            {"type": "text", "text": f"Standard step text: {standard_step['text']}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{standard_base64}"}} if standard_base64 else None,
            {"type": "text", "text": f"Actual step text: {actual_step['text']}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{actual_base64}"}} if actual_base64 else None
        ]}
    ]
    # Remove None entries
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], list):
            msg["content"] = [c for c in msg["content"] if c]

    response = client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4v"
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def main():
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_APIKEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAIAPI_VERSION")
    )

    # Example step data
    steps_json = [
        {
            "step_number": 1,
            "standard": {"text": "Open Control Panel", "image_path": "step1.png"},
            "actual": {"text": "Open Control Panel", "image_path": "step1_actual.png"}
        },
        {
            "step_number": 2,
            "standard": {"text": "Click 'Network and Internet'", "image_path": "step2.png"},
            "actual": {"text": "Click 'Network and Network'", "image_path": "step2_actual.png"}
        }
    ]

    results = []
    for step in steps_json:
        res = check_step(client, step["step_number"], step["standard"], step["actual"])
        results.append({"step_number": step["step_number"], "result": res})

    # Output results as JSON
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
