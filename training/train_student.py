from models.teacher import load_teacher_model
from models.student import StudentWrapper

from data.preprocessing_data import dataset_preprocessing
from datasets import Dataset
import torch
import torch.nn.functional as F
import wandb


def encoding_english_sentences():
    teacher_model = load_teacher_model()
    parallel_data = dataset_preprocessing()

    english_sentences = [pair[0] for pair in parallel_data]
    german_sentences = [pair[1] for pair in parallel_data]

    teacher_embeddings = teacher_model.encode(english_sentences, convert_to_tensor=False).tolist()

    return teacher_embeddings, german_sentences, english_sentences


def dataloader_creation(english_sentences, german_sentences, teacher_embeddings, config):
    BATCH_SIZE = config["training_args"]["batch_size"]
    dataset = Dataset.from_dict({
        "en": english_sentences,
        "de": german_sentences,
        "teacher_emb": teacher_embeddings
    })

    dataset.set_format(type='torch')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def prepare_dataset(config):
    """Full pipeline from raw tuples to Hugging Face Dataset."""
    teacher_embeddings, german_sentences, english_sentences = encoding_english_sentences()
    dataset = dataloader_creation(english_sentences, german_sentences, teacher_embeddings, config)

    return dataset


def distillation_loss(student_emb, teacher_emb):
    return F.mse_loss(student_emb, teacher_emb)


def training_loop(config, dataloader):
    # hyperparameters:
    LR = float(config['training_args']['learning_rate'])
    NUM_EPOCHS = config['training_args']['num_train_epochs']
    BATCH_SIZE = config['training_args']['batch_size']

    student_model = StudentWrapper()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)

    wandb.login(key='652f29755fbc034f857f7a0a6eec650bc13d5530')
    wandb.init(
        project="multilingual-distillation",
        config={
            "model": "paraphrase-xlm-r-multilingual-v1",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS
        }
    )

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_cos_sim = 0
        for batch_idx, batch in enumerate(dataloader):
            en_sentences = batch['en']
            de_sentences = batch['de']
            teacher_emb = batch['teacher_emb'].to(device)  # [batch_size, emb_dim]

            optimizer.zero_grad()

            # Get embeddings from student
            student_emb_en = student_model(en_sentences)
            student_emb_de = student_model(de_sentences)
            # Normalizing embeddings (optional but recommended)
            student_emb_en = F.normalize(student_emb_en, p=2, dim=1)
            student_emb_de = F.normalize(student_emb_de, p=2, dim=1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=1)

            # Compute loss (student EN + DE vs teacher EN)
            loss = distillation_loss(student_emb_en, teacher_emb) + distillation_loss(student_emb_de, teacher_emb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            wandb.log({"batch_loss": loss.item(), "batch_cosine_sim": F.cosine_similarity(student_emb_en, student_emb_de).mean().item()})

            epoch_loss += loss.item()
            epoch_cos_sim += F.cosine_similarity(student_emb_en, student_emb_de).mean().item()

        # log cosine similarity between English and German embeddings PER EPOCH
        avg_loss = epoch_loss / len(dataloader)
        avg_cos_sim = epoch_cos_sim / len(dataloader)
        wandb.log({"epoch_loss": avg_loss, "epoch_cosine_sim": avg_cos_sim})
        print(f"------EPOCH {epoch+1}-----------------")
        print(f"epoch_loss: {avg_loss}, cosine_sim: {avg_cos_sim}")

    wandb.finish()

