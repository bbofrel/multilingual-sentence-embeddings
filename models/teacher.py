import os
import yaml
from sentence_transformers import SentenceTransformer
from typing import Optional

# if not arg, then config, if not config-default
def load_teacher_model(model_name: Optional[str] = None):
    default_name = 'sentence-transformers/all-MiniLM-L6-v2'
    if model_name is None:
        try:
            with open("configs/sample.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_name = cfg.get("models", {}).get("teacher", default_name)
        except FileNotFoundError:
            model_name = default_name
    print(f"Teacher model: {model_name}")
    return SentenceTransformer(model_name)
