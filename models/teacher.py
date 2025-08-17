from sentence_transformers import SentenceTransformer


def load_teacher_model():
    teacher_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return teacher_model
