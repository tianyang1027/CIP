import json
from llm.client_manager import ClientManager
from enums.issue_enum import IssueEnum
from utils.file_utils import load_prompt


def check_steps_with_image_matching(steps_json, issue_type, judge_comment):

    PROMPT_FILES = {
        IssueEnum.FEATURE_NOT_FOUND.value: "llm/prompts/image_feature_not_found_prompt.txt",
        IssueEnum.NO_ISSUE_FOUND.value: "llm/prompts/image_no_issue_found_prompt.txt",
        IssueEnum.ISSUE_FOUND.value: "llm/prompts/image_issue_found_prompt.txt",
    }

    # Load the system prompt based on issue type
    system_prompt = load_prompt(PROMPT_FILES.get(issue_type))

    if issue_type in [IssueEnum.ISSUE_FOUND.value, IssueEnum.FEATURE_NOT_FOUND.value]:
        system_prompt = system_prompt.replace("{judge_comment}", judge_comment)

    # Prepare structured user content with 'type' fields
    user_content_structured = []
    
    # Add step relationship guidance BEFORE individual steps
    # This helps LLM understand prep-verification pairs and apply forward-scanning
    user_content_structured.append({
        "type": "text",
        "text": f"[STEP METADATA] Total steps in this case: {len(steps_json)}. Please identify preparation steps (e.g., 'Select language', 'Set option') and their corresponding verification steps (e.g., 'Verify X is playing', 'Verify Y state'). Use forward-scanning: evaluate verification steps first, then synchronize preparation step results with verification results. Do NOT judge preparation steps based on their weak screenshots alone; always check for corresponding verification steps."
    })
    
    for i, step in enumerate(steps_json):

        # Step header (helps GPT maintain structure and stability)
        user_content_structured.append({
            "type": "text",
            "text": f"=== Step {step['step_number']} ==="
        })

        # Standard text (always provided)
        user_content_structured.append({
            "type": "text",
            "text": f"Standard Text: {step['standard_text']}"
        })

        # Multi-action flag to guide judgment strictly
        if step.get("multi_action"):
            user_content_structured.append({
                "type": "text",
                "text": "Multi-Action: true — If current screenshot shows SOME sub-actions completed but others missing/unclear, mark NeedDiscussion (partial completion). Mark Correct only if ALL required sub-actions/final states are visible and complete. Mark Incorrect only if action is clearly failed/attempted but failed, or wrong page/content."
            })
        
        # Identify if this is a preparation or verification step
        step_text = step.get('standard_text', '').lower()
        if any(prep in step_text for prep in ['select', 'set ', 'choose', 'click', 'configure', 'enable', 'disable']):
            user_content_structured.append({
                "type": "text",
                "text": f"[STEP TYPE] Step {step['step_number']} appears to be a PREPARATION/CONFIGURATION step. Scan all subsequent steps for a corresponding VERIFICATION step. If found, derive this step's result from the verification result (forward-scanning rule)."
            })
        elif any(verif in step_text for verif in ['verify', 'confirm', 'check']):
            user_content_structured.append({
                "type": "text",
                "text": f"[STEP TYPE] Step {step['step_number']} is a VERIFICATION step. Evaluate this first if there is a preceding preparation step that depends on it. Verification steps MUST show the final outcome/state in their own screenshot; language selection flyout alone is NOT sufficient."
            })

        # Standard image (may not exist)
        if step.get("standard_image_url"):
            user_content_structured.append({
                "type": "text",
                "text": "Standard Image:"
            })
            user_content_structured.append({
                "type": "image_url",
                "image_url": {"url": step["standard_image_url"]}
            })
        else:
            user_content_structured.append({
                "type": "text",
                "text": "No Standard Image Provided."
            })

        # Actual image (may not exist)
        if step.get("actual_image_url"):
            user_content_structured.append({
                "type": "text",
                "text": "Actual Image:"
            })
            user_content_structured.append({
                "type": "image_url",
                "image_url": {"url": step["actual_image_url"]}
            })
        else:
            user_content_structured.append({
                "type": "text",
                "text": "No Actual Image Provided."
            })

        # IMPORTANT: Include next step's actual image for context
        # This allows verification of whether an action was completed, even if not visible in current step
        # Example: Step 2 says "open app X", Step 2 screenshot doesn't show it, but Step 3 shows app X is open
        # This proves Step 2's action was executed successfully
        if i + 1 < len(steps_json):
            next_step = steps_json[i + 1]
            if next_step.get("actual_image_url"):
                user_content_structured.append({
                    "type": "text",
                    "text": f"[CONTEXT FOR VERIFICATION] Next Step ({next_step['step_number']}) Actual Image - Use this to verify if Step {step['step_number']}'s action was completed:"
                })
                user_content_structured.append({
                    "type": "image_url",
                    "image_url": {"url": next_step["actual_image_url"]}
                })


    # Call GPT model
    content = ClientManager().chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_structured},
        ],
        model="gpt-5.1",
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

    # Enforce short-circuit: if any step returns NeedDiscussion, stop evaluating further steps
    step_results = parsed.get("step_results", [])
    short_circuit_index = None
    for idx, step in enumerate(step_results):
        res = (step or {}).get("result")
        if res == "NeedDiscussion":
            short_circuit_index = idx
            break

    if short_circuit_index is not None:
        step_results = step_results[: short_circuit_index + 1]
        final = {
            "final_result": "NeedDiscussion",
            "reason": step_results[short_circuit_index].get("reason", "A step requires NeedDiscussion; per rules, stop further evaluation.")
        }
        return {"step_results": step_results, "final_summary": final}

    # Validate final_summary when no NeedDiscussion encountered
    final = parsed.get("final_summary", {})
    final_result = final.get("final_result")
    if final_result not in {"Correct", "Incorrect", "Spam", "NeedDiscussion"}:
        final_result = "NeedDiscussion"
        final_reason = "Model returned unknown final_result, manual review required"
        final = {"final_result": final_result, "reason": final_reason}

    return {"step_results": step_results, "final_summary": final}


def compare_operations(standard_steps, actual_steps, issue_type, judge_comment):

    steps_json = build_steps_json(standard_steps, actual_steps)
    result = check_steps_with_image_matching(steps_json, issue_type, judge_comment)
    return result


def build_steps_json(standard_steps, actual_steps):
    """
    Build a simplified steps JSON containing only standard text and actual step image.
    Output format:
    [
        {"step_number": 1, "standard_text": "<text>", "standard_image_url": "<url>", "actual_image_url": "<url>"},
        ...
    ]
    """
    steps_json = []
    total_steps = max(len(standard_steps), len(actual_steps))
    import re
    def is_multi_action(text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        if re.search(r"\b\d+\s+(websites|tabs|accounts|items|pages)\b", t):
            return True
        if "," in t and any(k in t for k in ["gmail", "facebook", "twitter", "instagram", "linkedin"]):
            return True
        if re.search(r"\b(open|browse|search|login|log in|add|click|select)\b.*\band\b.*\b(open|browse|search|login|log in|add|click|select)\b", t):
            return True
        if re.search(r"\b(open|login|log in)\b[^\n]*,\s*[^\n]*,", t):
            return True
        return False

    for i in range(total_steps):
        step_number = i + 1
        standard_text = standard_steps[i]["text"] if i < len(standard_steps) else ""
        standard_image_url = (
            standard_steps[i].get("img") if i < len(standard_steps) else None
        )
        actual_image_url = actual_steps[i].get("img") if i < len(actual_steps) else None

        step = {"step_number": step_number, "standard_text": standard_text}
        step["multi_action"] = is_multi_action(standard_text)
        if standard_image_url:
            step["standard_image_url"] = standard_image_url

        if actual_image_url:
            step["actual_image_url"] = actual_image_url

        steps_json.append(step)

    return steps_json
