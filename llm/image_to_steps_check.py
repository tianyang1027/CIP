import os
import base64
from io import BytesIO
from PIL import Image
from llm.client_manager import ClientManager
import json



def check_steps_final_summary(steps_json):
    """
    Ask the model to internally evaluate each step but return ONLY a final_summary JSON object:
    {
      "final_result": "Correct" | "Incorrect" | "Spam" | "NeedDiscussion",
      "reason": "<summary reason>"
    }
    Priority for final_result: Spam > NeedDiscussion > Incorrect > Correct
    """

    system_prompt = (
        "You are an intelligent computer operation step checking assistant. "
        "You must carefully compare a series of computer operation steps performed by a user against the given standard steps. "
        "Each step includes a text description and may include an image for visual verification.\n\n"
        "Important instructions (READ CAREFULLY):\n"
        "1) Analyze EACH step in detail (text and image) and decide internally whether it is Correct, Incorrect, Spam, or NeedDiscussion. "
        "However, DO NOT output the per-step judgments. Keep those evaluations internal only.\n"
        "2) After analyzing all steps, produce ONLY a single JSON object named `final_summary` describing the overall result.\n"
        "3) The overall decision rules (priority) are:\n"
        '   - If any step is Spam (completely unrelated or nonsensical) => final_result = "Spam".\n'
        '   - Else if any step is NeedDiscussion (ambiguous / requires human review) => final_result = "NeedDiscussion".\n'
        '   - Else if any step is Incorrect => final_result = "Incorrect".\n'
        '   - Else (all steps are Correct) => final_result = "Correct".\n'
        "4) The final JSON object MUST have EXACTLY this shape and NOTHING else (no extra text):\n"
        "{\n"
        '  "final_summary": {\n'
        '    "final_result": "Correct" | "Incorrect" | "Spam" | "NeedDiscussion",\n'
        '    "reason": "<A concise explanation for the final_result>"\n'
        "  }\n"
        "}\n\n"
        "Additional guidance:\n"
        "- If an earlier step is incorrect, you must still analyze later steps and then apply the priority rules above to reach the final result.\n"
        "- The reason should summarize the main cause for the final decision (mention spam/ambiguity/which step(s) cause it, but keep it concise).\n"
        "- OUTPUT ONLY valid JSON (no markdown, no commentary, no extra fields)."
    )

    # Build compact user content describing steps (one-shot list style)
    user_content = []
    for step in steps_json:
        step_num = step.get("step_number")
        std_text = step["standard"].get("text", "")
        act_text = step["actual"].get("text", "")
        std_image_path = step["standard"].get("image_path")
        act_image_path = step["actual"].get("image_path")

        # Add a compact, clear description for each step
        user_content.append({"type": "text", "text": f"StepNumber: {step_num}"})
        
        user_content.append({"type": "text", "text": f"StandardText: {std_text}"})
        if std_image_path:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": std_image_path},
                }
            )

        user_content.append({"type": "text", "text": f"ActualText: {act_text}"})
        if act_image_path:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": act_image_path},
            }
        )

    # Call the model
    content = ClientManager().chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model="gpt-4.1",
        max_tokens=1500,
        temperature=0.0,
        top_p=1.0,
        timeout=120,
    )

    # Try to parse JSON and extract final_summary
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to recover JSON substring if model prepended/ appended whitespace or backticks (best effort)
        # Attempt to find first "{" and last "}" and parse that substring
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            parsed = json.loads(content[start:end])
        except Exception:
            # Give helpful debug output but still return an error-shaped summary
            print("⚠️ Warning: model output is not valid JSON. Raw output:")
            print(content)
            return {
                "final_summary": {
                    "final_result": "NeedDiscussion",
                    "reason": "Model output was not valid JSON; manual review required. See raw output.",
                }
            }

    # Normalize extracted final_summary if necessary
    # Cases handled:
    # 1) parsed is dict with 'final_summary' key
    # 2) parsed itself is the final_summary dict with final_result key
    # 3) parsed is a list or nested structure containing final_summary
    final = None
    if isinstance(parsed, dict):
        if "final_summary" in parsed and isinstance(parsed["final_summary"], dict):
            final = parsed["final_summary"]
        elif "final_result" in parsed:
            final = {
                "final_result": parsed.get("final_result"),
                "reason": parsed.get("reason", ""),
            }
        else:
            # maybe the model returned a wrapper like [{"final_summary": {...}}]
            # fall through to list handling by wrapping parsed into list
            parsed_list = [parsed]
            parsed = parsed_list

    if final is None and isinstance(parsed, list):
        # search list items for final_summary
        for item in parsed:
            if isinstance(item, dict):
                if "final_summary" in item and isinstance(item["final_summary"], dict):
                    final = item["final_summary"]
                    break
                if "final_result" in item:
                    final = {
                        "final_result": item.get("final_result"),
                        "reason": item.get("reason", ""),
                    }
                    break

    if final is None:
        # fallback: cannot find final_summary
        print("⚠️ Could not find 'final_summary' in model output. Raw parsed JSON:")
        print(parsed)
        return {
            "final_summary": {
                "final_result": "NeedDiscussion",
                "reason": "Model did not return a recognized final_summary structure; manual review required.",
            }
        }

    # Validate final fields
    final_result = final.get("final_result")
    reason = final.get("reason", "")

    if final_result not in {"Correct", "Incorrect", "Spam", "NeedDiscussion"}:
        return {
            "final_summary": {
                "final_result": "NeedDiscussion",
                "reason": "Model returned unknown final_result value; manual review required.",
            }
        }

    return {"final_summary": {"final_result": final_result, "reason": reason}}


def compare_operations(standard_steps, actual_steps):
    steps_json = build_steps_json(standard_steps, actual_steps)
    result = check_steps_final_summary(steps_json)
    return result


def build_steps_json(standard_steps, actual_steps):
    steps_json = []
    total_steps = max(len(standard_steps), len(actual_steps))

    for i in range(total_steps):
        standard = (
            standard_steps[i] if i < len(standard_steps) else {"text": "", "img": None}
        )
        actual = actual_steps[i] if i < len(actual_steps) else {"text": "", "img": None}

        steps_json.append(
            {
                "step_number": i + 1,
                "standard": {"text": standard["text"], "image_path": standard["img"]},
                "actual": {"text": actual["text"], "image_path": actual["img"]},
            }
        )

    return steps_json
