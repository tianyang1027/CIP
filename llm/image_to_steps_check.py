import json
from llm.client_manager import ClientManager
from enums.issue_enum import IssueEnum
from utils.file_utils import load_prompt


def check_steps_with_image_matching(steps_json, issue_type):

    PROMPT_FILES = {
        IssueEnum.FEATURE_NOT_FOUND.value: "llm/prompts/image_feature_not_found_prompt.txt",
        IssueEnum.NO_ISSUE_FOUND.value: "llm/prompts/image_no_issue_found_prompt.txt",
        IssueEnum.ISSUE_FOUND.value: "llm/prompts/image_issue_found_prompt.txt",
    }

    # Load the system prompt based on issue type
    system_prompt = load_prompt(PROMPT_FILES.get(issue_type))

    # Prepare structured user content with 'type' fields
    user_content_structured = []
    for step in steps_json:
        # standard step text
        user_content_structured.append(
            {
                "type": "text",
                "text": f"Step {step['step_number']} standard_text: {step['standard_text']}",
            }
        )
        # actual step image
        if step.get("actual_image_url"):
            user_content_structured.append(
                {"type": "image_url", "image_url": {"url": step["actual_image_url"]}}
            )

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
                    "reason": "Model output cannot be parsed, manual review required",
                },
            }

    # Validate final_summary
    final = parsed.get("final_summary", {})
    final_result = final.get("final_result")
    if final_result not in {"Correct", "Incorrect", "Spam", "NeedDiscussion"}:
        final_result = "NeedDiscussion"
        final_reason = "Model returned unknown final_result, manual review required"
        final = {"final_result": final_result, "reason": final_reason}

    return {"step_results": parsed.get("step_results", []), "final_summary": final}


def compare_operations(standard_steps, actual_steps, issue_type):

    steps_json = build_steps_json(standard_steps, actual_steps)
    result = check_steps_with_image_matching(steps_json, issue_type)
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
