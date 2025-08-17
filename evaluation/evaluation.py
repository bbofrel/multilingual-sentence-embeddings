from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch

def load_tatoeba (limit=1000):
    ttb=load_dataset("tatoeba", "eng-deu", split="test")
    ttb=ttb.select(range(min(limit, len(ttb))))
    en = [r["sourceSentence"] for r in ttb]
    de = [r["targetSentence"] for r in ttb]
    return en, de

def evaluation_tatoeba(model_path="models/student_en_de", limit=1000):
    model=SentenceTransformer(model_path)
    en,de=load_tatoeba(limit)