import numpy as np
import torch
import yaml
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, util

from models.student import StudentWrapper

MONO_TRACKS = ["en-en", "de-de"]
CROSS_TRACKS = ["en-de"]


def load_sts17(lang_pair):
    """ loads and filters the STS17 cross-lingual dataset for a specific language pair """
    dataset = load_dataset("mteb/sts17-crosslingual-sts", split="test")
    keep = ["en-de", "de-en"] if lang_pair == "en-de" else [lang_pair]  # keeping both directions
    filtered_dataset = dataset.filter(lambda row: row.get("language") in keep)  # filtering rows by the language column
    if len(filtered_dataset) == 0:
        uniq = sorted(set(dataset["language"])) if "language" in dataset.column_names else []
        raise ValueError(f"No data for {lang_pair}. Available 'language' values: {uniq}")
    s1 = [row["sentence1"] for row in filtered_dataset]
    s2 = [row["sentence2"] for row in filtered_dataset]
    y = [float(row["score"]) for row in filtered_dataset]
    return s1, s2, y


def encode_teacher(model_name, texts, config, batch_size=None, normalize=True):
    """ encodes text inputs using a pretrained SentenceTransformer (teacher model) """
    if batch_size is None:
        batch_size = config['training_args']['batch_size']
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True, batch_size=batch_size,
                       normalize_embeddings=normalize)
    return embeddings


def encode_student(checkpoint_path, texts, config, batch_size=None, normalize=True, device=None):
    """ encodes text inputs using a custom student model from a checkpoint """
    if batch_size is None:
        batch_size = config['training_args']['batch_size']
    state_dict_weights = torch.load(checkpoint_path, map_location="cpu")
    out_dim = None
    for k in ["proj.weight", "projection.weight", "proj.fc.weight"]:
        if k in state_dict_weights and hasattr(state_dict_weights[k], "shape"):
            out_dim = state_dict_weights[k].shape[0]
            break
    student = StudentWrapper(model_name=config.get('models').get('student'))
    if out_dim is not None and hasattr(student, "set_output_dim"):
        student.set_output_dim(out_dim)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device)
    student.load_state_dict(state_dict_weights, strict=False)
    student.eval()
    batch_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            e = student(batch).to("cpu")
            batch_embs.append(e)
    student_embeddings = torch.vstack(batch_embs)
    if normalize:
        student_embeddings = torch.nn.functional.normalize(student_embeddings, p=2, dim=1)
    return student_embeddings


def eval_tracks(encoder_kind, encoder_ref, tracks, config):
    """ evaluates either the teacher or student encoder on a set of tracks.
        only students models were evaluated for the current study """
    results = {}  # storing the Spearman correlation for each track
    for lang_pair in tracks:
        s1, s2, gold = load_sts17(lang_pair)
        if encoder_kind == "teacher":
            e1 = encode_teacher(encoder_ref, s1, config)
            e2 = encode_teacher(encoder_ref, s2, config)
        else:
            e1 = encode_student(encoder_ref, s1, config)
            e2 = encode_student(encoder_ref, s2, config)
        # cosine_sims: 1D array of predicted similarities for i-th x i-th sentence
        cosine_sims = util.cos_sim(e1, e2).diagonal().cpu().numpy()
        correlation = spearmanr(cosine_sims, np.asarray(gold, float)).correlation * 100.0
        results[lang_pair] = float(correlation)
        print(f"{lang_pair:>6}: {correlation:5.1f}")
    avg = float(np.mean(list(results.values()))) if results else float("nan")
    print(f"{'Avg.':>6}: {avg:5.1f}")
    return results, avg


if __name__ == "__main__":
    with open("configs/sample.yaml", "r") as f:
        config = yaml.safe_load(f)
    checkpoint_path = "models/checkpoints/student_xlmr_distilled.pt"
    print("Mono_tracks (EN-EN):")
    eval_tracks(config['evaluation']['mode'], checkpoint_path, MONO_TRACKS, config)
    print("\nCross_track (EN-DE):")
    eval_tracks(config['evaluation']['mode'], checkpoint_path, CROSS_TRACKS, config)
