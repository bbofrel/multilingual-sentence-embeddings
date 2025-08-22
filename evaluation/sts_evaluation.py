import argparse, torch, numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, util
from models.student import StudentWrapper

MONO_TRACKS  = ["en-en", "de-de"]
CROSS_TRACKS = ["en-de"]

def load_sts17(lang_pair):
    ds = load_dataset("mteb/sts17-crosslingual-sts", split="test")
    keep = ["en-de", "de-en"] if lang_pair == "en-de" else [lang_pair]
    sub = ds.filter(lambda r: r.get("language") in keep)
    if len(sub) == 0:
        uniq = sorted(set(ds["language"])) if "language" in ds.column_names else []
        raise ValueError(f"No data for {lang_pair}. Available 'language' values: {uniq}")
    s1 = [r["sentence1"] for r in sub]
    s2 = [r["sentence2"] for r in sub]
    y  = [float(r["score"]) for r in sub]
    return s1, s2, y

def encode_sbert(model_name, texts, batch_size=128, normalize=True):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_tensor=True, batch_size=batch_size,
                       normalize_embeddings=normalize)
    return emb

def encode_student(ckpt_path, texts, batch_size=64, normalize=True, device=None):
    sd = torch.load(ckpt_path, map_location="cpu")
    out_dim = None
    for k in ["proj.weight", "projection.weight", "proj.fc.weight"]:
        if k in sd and hasattr(sd[k], "shape"):
            out_dim = sd[k].shape[0]
            break
    student = StudentWrapper(model_name="xlm-roberta-base")
    if out_dim is not None and hasattr(student, "set_output_dim"):
        student.set_output_dim(out_dim)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device)
    student.load_state_dict(sd, strict=False)
    student.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            e = student(batch).to("cpu")
            embs.append(e)
    emb = torch.vstack(embs)
    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb

def eval_tracks(encoder_kind, encoder_ref, tracks):
    results = {}
    for lp in tracks:
        s1, s2, gold = load_sts17(lp)
        if encoder_kind == "sbert":
            e1 = encode_sbert(encoder_ref, s1)
            e2 = encode_sbert(encoder_ref, s2)
        else:
            e1 = encode_student(encoder_ref, s1)
            e2 = encode_student(encoder_ref, s2)
        sims = util.cos_sim(e1, e2).diagonal().cpu().numpy()
        rho  = spearmanr(sims, np.asarray(gold, float)).correlation * 100.0
        results[lp] = float(rho)
        print(f"{lp:>6}: {rho:5.1f}")
    avg = float(np.mean(list(results.values()))) if results else float("nan")
    print(f"{'Avg.':>6}: {avg:5.1f}")
    return results, avg

if __name__ == "__main__":
    ckpt_path = "models/student_xlmr_distilled.pt"
    print("Mono_tracks (EN-EN, DE-DE):")
    eval_tracks("student", ckpt_path, MONO_TRACKS)
    print("\nCross_track (EN-DE):")
    eval_tracks("student", ckpt_path, CROSS_TRACKS)

