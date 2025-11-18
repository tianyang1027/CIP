from pathlib import Path

def load_prompt(filename: str) -> str:
    """
    Load the content of a prompt text file and return it as a string.

    :param filename: Path to the prompt txt file.
    :return: The prompt content as a string.
    :raises FileNotFoundError: If the file does not exist.
    """
    prompt_path = Path(filename)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {filename}")
    
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()