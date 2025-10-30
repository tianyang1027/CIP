import json
from typing import List, Dict

def load_json(file_path: str):
    """Load JSON file from disk"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
def compare_actions_short_circuit(standard_steps: List[Dict], user_actions: List[Dict]) -> str:
    """
    Compare standard steps and user actions from video.
    Stop at the first incorrect step.
    Return a single string: "Correct" or "Wrong: Step X [Title] - Reason".
    """
    user_action_names = [a["action"] for a in user_actions]
    used_indices = set()

    for step in standard_steps:
        expected = step.get("expected_actions", [])
        matched_actions = []
        missed_actions = []

        for action in expected:
            try:
                index = next(
                    i for i, a in enumerate(user_action_names)
                    if a == action and i not in used_indices
                )
                matched_actions.append(action)
                used_indices.add(index)
            except StopIteration:
                missed_actions.append(action)

        if missed_actions:
            # Stop immediately on first error
            reason = f"Missing actions: {missed_actions}; Detected: {matched_actions}"
            return f"Wrong: Step {step.get('step')} [{step.get('title')}] - {reason}"

    # If all steps are correct
    return "Correct"