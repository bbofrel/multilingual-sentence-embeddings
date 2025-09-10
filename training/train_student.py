import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset
from sklearn.model_selection import train_test_split

from data.preprocessing_data import dataset_preprocessing
from models.student import StudentWrapper
from models.teacher import load_teacher_model


def encoding_english_sentences(config, version='full'):
    teacher_model = load_teacher_model(
        config.get("models").get("teacher"))
    dataset_size = config.get("dataset").get("dataset_size")
    parallel_data = dataset_preprocessing(version=dataset_size)
    parallel_data = list(dict.fromkeys(parallel_data))
    english_sentences = [pair[0] for pair in parallel_data]
    german_sentences = [pair[1] for pair in parallel_data]

    teacher_embeddings = teacher_model.encode(english_sentences, convert_to_tensor=False).tolist()

    return teacher_embeddings, german_sentences, english_sentences


def dataloader_creation(english_sentences, german_sentences, teacher_embeddings, config, shuffle=True):
    BATCH_SIZE = config["training_args"]["batch_size"]
    dataset = Dataset.from_dict({
        "en": english_sentences,
        "de": german_sentences,
        "teacher_emb": teacher_embeddings
    })

    dataset.set_format(type='torch')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return dataloader


def prepare_dataset(config):
    # train and dev dataloaders
    teacher_embeddings, german_sentences, english_sentences = encoding_english_sentences(config)

    # train vs dev split (10%)
    en_train, en_dev, de_train, de_dev, t_train, t_dev = train_test_split(
        english_sentences,
        german_sentences,
        teacher_embeddings,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    print(f"Split -> train={len(en_train)} | dev={len(en_dev)}")

    train_loader = dataloader_creation(en_train, de_train, t_train, config, shuffle=True)
    dev_loader = dataloader_creation(en_dev, de_dev, t_dev, config, shuffle=False)
    return train_loader, dev_loader


def distillation_loss(student_emb, teacher_emb):
    return F.mse_loss(student_emb, teacher_emb)


def training_loop(config, train_loader, dev_loader):
    LR = float(config['training_args']['learning_rate'])
    NUM_EPOCHS = config['training_args']['num_train_epochs']
    BATCH_SIZE = config['training_args']['batch_size']

    student_model = StudentWrapper(model_name=config.get("models").get(
        "student", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    print(f"Student model: {config.get('models').get('student')}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(device)

    first_batch = next(iter(train_loader))
    teacher_dim = int(first_batch["teacher_emb"].shape[1])
    student_model.set_output_dim(teacher_dim)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)

    wandb.login(key='652f29755fbc034f857f7a0a6eec650bc13d5530')
    wandb.init(project="multilingual-distillation",
               config={"model": "paraphrase-xlm-r-multilingual-v1", "lr": LR, "batch_size": BATCH_SIZE,
                       "epochs": NUM_EPOCHS})

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        epoch_loss, epoch_cos_sim, epoch_n = 0.0, 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            en_sentences = batch['en']
            de_sentences = batch['de']
            teacher_emb = batch['teacher_emb'].to(device)

            optimizer.zero_grad()
            student_emb_en = student_model(en_sentences)
            student_emb_de = student_model(de_sentences)

            student_emb_en = F.normalize(student_emb_en, p=2, dim=1)
            student_emb_de = F.normalize(student_emb_de, p=2, dim=1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=1)

            loss = distillation_loss(student_emb_en, teacher_emb) + distillation_loss(student_emb_de, teacher_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = teacher_emb.size(0)  # [batch_size, emb_dim]
            epoch_n += batch_size

            wandb.log({"batch_loss": loss.item(),
                       "batch_cosine_sim": F.cosine_similarity(student_emb_en, student_emb_de).mean().item()})
            epoch_loss += loss.item() * batch_size
            epoch_cos_sim += F.cosine_similarity(student_emb_en, student_emb_de).mean().item() * batch_size

        avg_loss = epoch_loss / epoch_n
        avg_cos = epoch_cos_sim / epoch_n
        wandb.log({"epoch_loss": avg_loss, "epoch_cosine_sim": avg_cos})
        print(f"------EPOCH {epoch + 1}-----------------")
        print(f"epoch_loss: {avg_loss:.6f}, cosine_sim: {avg_cos:.4f}")

        student_model.eval()
        dev_mse_sum, dev_cos_sum, dev_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in dev_loader:
                de_sentences = batch['de']
                teacher_emb = batch['teacher_emb'].to(device)  # teacher's english

                student_de_val = student_model(de_sentences)  # student's german
                # normalizing
                student_de_val = F.normalize(student_de_val, p=2, dim=1)
                teacher_emb = F.normalize(teacher_emb, p=2, dim=1)

                batch_size = teacher_emb.size(0)  # [batch_size, emb_dim]

                mse_per_sample = F.mse_loss(student_de_val, teacher_emb, reduction='none').mean(dim=1)
                dev_mse_sum += mse_per_sample.sum().item()
                cos_per_sample = F.cosine_similarity(student_de_val, teacher_emb, dim=1)
                dev_cos_sum += cos_per_sample.sum().item()

                dev_n += batch_size

        dev_mse = dev_mse_sum / max(1, dev_n)
        dev_cos = dev_cos_sum / max(1, dev_n)
        print(f"validation MSE: {dev_mse:.6f} | dev_cosine_sim: {dev_cos:.4f}")
        wandb.log({"dev_mse": dev_mse, "dev_cosine_sim": dev_cos})

        torch.save(student_model.state_dict(), "models/student_xlmr_distilled.pt")

    wandb.finish()
