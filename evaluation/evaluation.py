from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import yaml
from sentence_transformers.util import normalize_embeddings


def load_tatoeba (limit=1000):
    ttb=load_dataset("Helsinki-NLP/tatoeba_mt", "eng-deu", split="test", trust_remote_code=True).select(range(limit))
    en = [r["source_sentence"] for r in ttb]
    de = [r["target_sentence"] for r in ttb]
    return en, de

def evaluation_tatoeba(model_path=None, limit=1000):
    if model_path is None:
        with open("configs/sample.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_path = cfg.get("models", {}).get(
            "student", "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        )
    print(f"Evaluation model: {model_path}")
    model = SentenceTransformer(model_path)
    en,de=load_tatoeba(limit)
    en_embd = model.encode(en, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    de_embd = model.encode(de, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    sims = util.cos_sim(en_embd, de_embd)
    acc = (sims.argmax(dim=1) == torch.arange(len(de_embd), device=sims.device)).float().mean().item()
    print(f"Tatoeba EN→DE retrieval@1 on {len(en)}: {acc:.3f}")


if __name__ == "__main__":
    evaluation_tatoeba("sentence-transformers/paraphrase-xlm-r-multilingual-v1", limit=500)
    #evaluation_tatoeba(limit=1000)