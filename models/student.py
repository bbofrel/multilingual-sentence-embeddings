from sentence_transformers import SentenceTransformer

def load_student(name: str = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"):
    return SentenceTransformer(name)