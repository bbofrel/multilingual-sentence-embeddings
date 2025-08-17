from models.teacher import load_teacher_model
from data.preprocessing_data import dataset_preprocessing


def encoding_english_sentences():
    teacher_model = load_teacher_model()
    parallel_data = dataset_preprocessing()

    english_sentences = [pair[0] for pair in parallel_data]
    german_sentences = [pair[1] for pair in parallel_data]

    teacher_embeddings = teacher_model.encode(english_sentences, convert_to_tensor=False).tolist()

    return teacher_embeddings, german_sentences

def dataloader_creation(german_sentences, teacher_embeddings, config):
    BATCH_SIZE = config["training_args"]["batch_size"]
    dataset = Dataset.from_dict({
        "german_sentence": german_sentences,
        "teacher_embedding": teacher_embeddings
    })

    dataset.set_format(type='torch', columns=['teacher_embedding'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def prepare_dataset(config):
    """Full pipeline from raw tuples to Hugging Face Dataset."""
    teacher_embeddings, german_sentences = encoding_english_sentences()
    dataset = dataloader_creation(german_sentences, teacher_embeddings, config)

    return dataset