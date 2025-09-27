from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import yaml

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.student import StudentWrapper
import torch.nn.functional as F


def load_tatoeba(limit=1000):
    """ loads a subset of the Tatoeba German–English dataset """
    ttb = load_dataset("Helsinki-NLP/tatoeba_mt", "deu-eng",
                       split="test", trust_remote_code=True, verification_mode="no_checks").select(range(limit))
    en = [r["targetString"] for r in ttb]
    de = [r["sourceString"] for r in ttb]
    return en, de

def evaluation_tatoeba(model_path=None, limit=1000):
    """ evaluates a teacher or student model on Tatoeba EN-DE retrieval """
    with open("../configs/sample.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if model_path is None:
        model_path = cfg.get("models", {}).get(
            "student", "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        )

    print(f"Evaluation model: {model_path}")
    en, de = load_tatoeba(limit)

    if model_path.endswith(".pt"):
        base_model_name = cfg.get("models", {}).get("student", "xlm-roberta-base")
        state = torch.load(model_path, map_location="cpu")
        student = StudentWrapper(model_name=base_model_name)
        student.load_state_dict(state, strict=False)
        student.eval()

        def encode_student(texts):
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(texts), 128):
                    batch = texts[i:i + 128]
                    emb = student(batch)
                    emb = F.normalize(emb, p=2, dim=1)
                    all_embs.append(emb)
            return torch.cat(all_embs, dim=0)

        en_embd = encode_student(en)
        de_embd = encode_student(de)
    else:
        model = SentenceTransformer(model_path)
        en_embd = model.encode(en, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
        de_embd = model.encode(de, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)

    sims_en_de = util.cos_sim(en_embd, de_embd)
    acc_en_de = (sims_en_de.argmax(dim=1) == torch.arange(len(de_embd), device=sims_en_de.device)).float().mean().item()

    sims_de_en = util.cos_sim(de_embd, en_embd)
    acc_de_en = (sims_de_en.argmax(dim=1) == torch.arange(len(en_embd), device=sims_de_en.device)).float().mean().item()

    avg = (acc_en_de + acc_de_en) / 2.0
    n = len(en)

    print(f"Tatoeba EN→DE @1 on {n}: {acc_en_de:.3f}")
    print(f"Tatoeba DE→EN @1 on {n}: {acc_de_en:.3f}")
    print(f"Avg.: {avg:.3f}")


if __name__ == "__main__":
    evaluation_tatoeba("../models/checkpoints/student_xlmr_distilled.pt", limit=1000)