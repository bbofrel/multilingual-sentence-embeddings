from models.teacher import load_teacher_model
from models.student import StudentWrapper

from data.preprocessing_data import dataset_preprocessing
from datasets import Dataset
import torch
import torch.nn.functional as F
import wandb


def encoding_english_sentences(config):
    teacher_model = load_teacher_model(
        config.get("models", {}).get("teacher"))
    parallel_data = dataset_preprocessing()

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
    #Train and dev dataloaders
    teacher_embeddings, german_sentences, english_sentences = encoding_english_sentences(config)

    #train vs dev split (10%)
    import random, numpy as np
    N = len(english_sentences)
    idx = list(range(N))
    random.Random(42).shuffle(idx)
    cut = max(1, int(0.10 * N))

    dev_idx   = np.array(idx[:cut])
    train_idx = np.array(idx[cut:])

    en_train = [english_sentences[i] for i in train_idx]
    de_train = [german_sentences[i] for i in train_idx]
    t_train  = [teacher_embeddings[i] for i in train_idx]

    en_dev = [english_sentences[i] for i in dev_idx]
    de_dev = [german_sentences[i] for i in dev_idx]
    t_dev  = [teacher_embeddings[i] for i in dev_idx]

    print(f"Split -> train={len(en_train)} | dev={len(en_dev)}")

    assert not (set(en_train) & set(en_dev))
    assert not (set(de_train) & set(de_dev))

    train_loader = dataloader_creation(en_train, de_train, t_train, config, shuffle=True)
    dev_loader   = dataloader_creation(en_dev,   de_dev,   t_dev,   config, shuffle=False)
    return train_loader, dev_loader

    return dataset


def distillation_loss(student_emb, teacher_emb):
    return F.mse_loss(student_emb, teacher_emb)


def training_loop(config, train_loader, dev_loader):
    LR = float(config['training_args']['learning_rate'])
    NUM_EPOCHS = config['training_args']['num_train_epochs']
    BATCH_SIZE = config['training_args']['batch_size']

    student_model = StudentWrapper(model_name=config.get("models", {}).get(
        "student", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    print(f"Student model: {config.get('models', {}).get('student')}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model.to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)

    wandb.login(key='652f29755fbc034f857f7a0a6eec650bc13d5530')
    wandb.init(project="multilingual-distillation",
               config={"model": "paraphrase-xlm-r-multilingual-v1", "lr": LR, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS})

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        epoch_loss = 0.0
        epoch_cos_sim = 0.0

        for batch_idx, batch in enumerate(train_loader):
            en_sentences = batch['en']
            de_sentences = batch['de']
            teacher_emb = batch['teacher_emb'].to(device)

            optimizer.zero_grad()
            student_emb_en = student_model(en_sentences)
            student_emb_de = student_model(de_sentences)

            student_emb_en = F.normalize(student_emb_en, p=2, dim=1)
            student_emb_de = F.normalize(student_emb_de, p=2, dim=1)
            teacher_emb    = F.normalize(teacher_emb,    p=2, dim=1)

            loss = distillation_loss(student_emb_en, teacher_emb) + distillation_loss(student_emb_de, teacher_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            wandb.log({"batch_loss": loss.item(),
                       "batch_cosine_sim": F.cosine_similarity(student_emb_en, student_emb_de).mean().item()})
            epoch_loss += loss.item()
            epoch_cos_sim += F.cosine_similarity(student_emb_en, student_emb_de).mean().item()

        avg_loss = epoch_loss / len(train_loader)
        avg_cos  = epoch_cos_sim / len(train_loader)
        wandb.log({"epoch_loss": avg_loss, "epoch_cosine_sim": avg_cos})
        print(f"------EPOCH {epoch+1}-----------------")
        print(f"epoch_loss: {avg_loss:.6f}, cosine_sim: {avg_cos:.4f}")

        # DEV
        student_model.eval()
        dev_mse_sum, dev_cos_sum, dev_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in dev_loader:
                de = batch['de']
                t  = batch['teacher_emb'].to(device)

                s_de = student_model(de)
                s_de = F.normalize(s_de, p=2, dim=1)
                t    = F.normalize(t,    p=2, dim=1)

                mse = F.mse_loss(s_de, t)
                cos = (s_de * t).sum(dim=1).mean()

                bs = t.size(0)
                dev_mse_sum += mse.item() * bs
                dev_cos_sum += cos.item() * bs
                dev_n += bs

        dev_mse = dev_mse_sum / max(1, dev_n)
        dev_cos = dev_cos_sum / max(1, dev_n)
        print(f"[DEV] mse: {dev_mse:.6f} | cos: {dev_cos:.4f}")
        wandb.log({"dev/mse": dev_mse, "dev/cos": dev_cos})

    wandb.finish()


