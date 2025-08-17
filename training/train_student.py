from models.teacher import load_teacher_model
from data.preprocessing_data import dataset_preprocessing


def encoding_english_sentences(dataset):
    teacher_model = load_teacher_model()
    parallel_data = dataset_preprocessing()

    english_sentences = [pair[0] for pair in parallel_data]
    assert isinstance(english_sentences, List)

    embeddings = teacher_model.encode(sentences, convert_to_tensor=True)
    return embeddings