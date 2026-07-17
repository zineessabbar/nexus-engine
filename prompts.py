# prompts.py
import yaml
from pathlib import Path

_prompts = None

def load_prompts() -> dict:
    global _prompts
    if _prompts is None:
        path = Path(__file__).parent / "prompts.yaml"
        with open(path, encoding="utf-8") as f:
            _prompts = yaml.safe_load(f)
    return _prompts

def get_prompt(name: str, **kwargs) -> str:
    prompts = load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' introuvable dans prompts.yaml")
    return prompts[name].format(**kwargs)