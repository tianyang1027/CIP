import json
from llm.client_manager import ClientManager

# just for no issue found
def check_steps_with_image_matching(steps_json):
    """
    GPT matches each actual step image against the standard step text.
    Output format:
    {
        "step_results": [
            {"step_number": 1, "result": "Correct"|"Incorrect"|"Spam"|"NeedDiscussion", "reason": "..."},
            ...
        ],
        "final_summary": {
            "final_result": "Correct"|"Incorrect"|"Spam"|"NeedDiscussion",
            "reason": "..."
        }
    }
    """

    system_prompt = (
        "You are an intelligent step checking assistant.\n"
        "Each step has a standard text and an actual step image (URL).\n"
        "Your goal is to determine whether each actual image matches the standard step text.\n\n"

        "IMPORTANT OVERRIDE RULE:\n"
        "- If the standard step describes **Microsoft Edge browser settings, configuration, or setup operations**, "
        "then this step must be automatically judged as **Correct**, without checking or comparing the actual image at all.\n"
        "- Completely ignore the screenshot for such steps.\n"
        "- These Edge browser setting/configuration steps include actions like:\n"
        "  - Opening Edge settings or preferences\n"
        "  - Configuring browser options\n"
        "  - Setting up profiles or accounts\n"
        "  - Adjusting security/privacy/extension settings\n"
        "  - Managing startup behavior or homepage\n"
        "  - Any other Edge browser configuration tasks\n"
        "- For these steps, always output:\n"
        '  result = "Correct"\n'
        "  reason = \"This is an Edge browser setting step, automatically considered correct as instructed.\"\n\n"

        "For all other non-browser-setting steps, follow the normal rules below:\n"
        "- Correct: The image clearly shows the step was performed as described.\n"
        "- Incorrect: The image deviates from the standard step.\n"
        "- Spam: The image is unrelated or nonsensical.\n"
        "- NeedDiscussion: Image is unclear or ambiguous.\n\n"

        "Final summary rules:\n"
        "- If any step is Spam → final_result = 'Spam'\n"
        "- Else if any step is NeedDiscussion → final_result = 'NeedDiscussion'\n"
        "- Else if any non-browser-setting step is Incorrect → final_result = 'Incorrect'\n"
        "- Else (all steps are Correct) → final_result = 'Correct'\n\n"

        "Output JSON strictly in this format:\n"
        "{\n"
        '  "step_results": [\n'
        '    {"step_number": 1, "result": "Correct"|"Incorrect"|"Spam"|"NeedDiscussion", "reason": "<reason>"},\n'
        '    ...\n'
        '  ],\n'
        '  "final_summary": {\n'
        '    "final_result": "Correct"|"Incorrect"|"Spam"|"NeedDiscussion",\n'
        '    "reason": "<final reason>"\n'
        '  }\n'
        "}\n\n"

        "Notes:\n"
        "1) Every step must include a result and a reason.\n"
        "2) Follow the OVERRIDE RULE strictly for Edge browser setting steps.\n"
        "3) Do not output anything except JSON."
    )

    # Prepare structured user content with 'type' fields
    user_content_structured = []
    for step in steps_json:
        # standard step text
        user_content_structured.append({
            "type": "text",
            "text": f"Step {step['step_number']} standard_text: {step['standard_text']}"
        })
        # actual step image
        if step.get("actual_image_url"):
            user_content_structured.append({
                "type": "image_url",
                "image_url": {"url": step["actual_image_url"]}
            })

    # Call GPT model
    content = ClientManager().chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_structured},
        ],
        model="gpt-4.1",
        max_tokens=2000,
        temperature=0.0,
        top_p=1.0,
        timeout=120,
    )

    # Parse JSON
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            parsed = json.loads(content[start:end])
        except Exception:
            print("Warning: model output is not valid JSON.")
            print(content)
            return {
                "step_results": [],
                "final_summary": {
                    "final_result": "NeedDiscussion",
                    "reason": "Model output cannot be parsed, manual review required"
                }
            }

    # Validate final_summary
    final = parsed.get("final_summary", {})
    final_result = final.get("final_result")
    if final_result not in {"Correct", "Incorrect", "Spam", "NeedDiscussion"}:
        final_result = "NeedDiscussion"
        final_reason = "Model returned unknown final_result, manual review required"
        final = {"final_result": final_result, "reason": final_reason}

    return {
        "step_results": parsed.get("step_results", []),
        "final_summary": final
    }


def compare_operations(standard_steps, actual_steps):
    steps_json = build_steps_json(standard_steps, actual_steps)
    result = check_steps_with_image_matching(steps_json)
    return result


def build_steps_json(standard_steps, actual_steps):
    """
    Build a simplified steps JSON containing only standard text and actual step image.
    Output format:
    [
        {"step_number": 1, "standard_text": "<text>", "actual_image_url": "<url>"},
        ...
    ]
    """
    steps_json = []
    total_steps = max(len(standard_steps), len(actual_steps))
    for i in range(total_steps):
        step_number = i + 1
        standard_text = standard_steps[i]["text"] if i < len(standard_steps) else ""
        actual_image_url = actual_steps[i].get("img") if i < len(actual_steps) else None

        step = {"step_number": step_number, "standard_text": standard_text}
        if actual_image_url:
            step["actual_image_url"] = actual_image_url

        steps_json.append(step)

    return steps_json
