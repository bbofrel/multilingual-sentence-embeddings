from sentence_transformers import SentenceTransformer


def load_teacher_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model
