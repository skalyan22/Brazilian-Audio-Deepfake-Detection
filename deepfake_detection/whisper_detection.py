# 1. SETUP

# !pip install -q transformers torch torchaudio datasets scikit-learn librosa soundfile tqdm
import numpy as np, torch, warnings
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve)
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/whisper-small"   # swap to whisper-base for faster / -medium for stronger
SR = 16000
print("Device:", DEVICE, "| Model:", MODEL_NAME)

# 2. LOAD THE DATASET

ds = load_dataset("AKCIT-Deepfake/BRSpeech-DF")
print(ds)
# Inspect one split's schema
split0 = list(ds.keys())[0]
print("\nColumns:", ds[split0].column_names)
print("\nExample:", {k: v for k, v in ds[split0][0].items() if k != 'audio'})


# 2a. MAP Columnns
# Adjust the variables below after inspecting the output above
# We need:
#   - an audio column,
#   - a label column (real vs fake),
#   - optionally a speaker/source column for a leakage-free split


# ---- EDIT THESE after inspecting the schema above ----
AUDIO_COL   = "audio"
LABEL_COL   = "label"        # e.g. "label", "class", "spoof"
SPEAKER_COL = None           # e.g. "speaker_id" / "source" / "tts_engine"; None -> random split
# ------------------------------------------------------

def normalize_label(v):
    """Return 1 for FAKE/spoof, 0 for REAL/bonafide."""
    s = str(v).strip().lower()
    if s in {"1","fake","spoof","spoofed","synthetic","tts","deepfake"}: return 1
    if s in {"0","real","bonafide","genuine","human"}: return 0
    # numeric fallback
    try: return int(float(s) != 0)
    except: raise ValueError(f"Unmapped label: {v!r}")

data = ds[split0].cast_column(AUDIO_COL, Audio(sampling_rate=SR))
labels_preview = [normalize_label(x) for x in data.select(range(min(200, len(data))))[LABEL_COL]]
print("Preview label balance (0=real,1=fake):", np.bincount(labels_preview))

# 3. Feature extraction with the frozen Whisper encoder
# For each clip we compute the log-mel input, run it through the encoder, and mean-pool
# the final hidden states into a fixed-length embedding
fe = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
encoder = WhisperModel.from_pretrained(MODEL_NAME).encoder.to(DEVICE).eval()

@torch.no_grad()
def embed(audio_array, sr=SR):
    feats = fe(audio_array, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
    out = encoder(feats).last_hidden_state          # (1, T, H)
    return out.mean(dim=1).squeeze(0).cpu().numpy() # (H,)

# Optional: subsample for a quick run. Set N=None to use the full split.
N = 1500
idx = range(len(data)) if N is None else range(min(N, len(data)))

X, y, groups = [], [], []
for i in tqdm(idx, desc="Embedding"):
    row = data[i]
    a = row[AUDIO_COL]["array"].astype(np.float32)
    if a.size == 0:
        continue
    X.append(embed(a))
    y.append(normalize_label(row[LABEL_COL]))
    groups.append(row[SPEAKER_COL] if SPEAKER_COL else i)

X = np.vstack(X); y = np.array(y); groups = np.array(groups)
print("Feature matrix:", X.shape, "| class balance:", np.bincount(y))

# 4. Leakage-aware train/test split
# If a speaker/source column is available we use `GroupShuffleSplit` so the same speaker or TTS
# engine never appears in both train and test — otherwise the classifier can cheat on voice
# identity instead of learning spoof artifacts.
from sklearn.model_selection import GroupShuffleSplit, train_test_split

if SPEAKER_COL is not None and len(np.unique(groups)) > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    print("Group-aware split (leakage-safe).")
else:
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                              stratify=y, random_state=42)
    print("Random stratified split (no group column — beware speaker leakage).")

Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
print("Train:", Xtr.shape, "Test:", Xte.shape)


# 5. Train the classifier
scaler = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
clf.fit(scaler.transform(Xtr), ytr)

scores = clf.predict_proba(scaler.transform(Xte))[:, 1]  # P(fake)
preds  = (scores >= 0.5).astype(int)

# 6. Evaluation (EER, balanced accuracy, AUC)
def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2, thr[i]

eer, eer_thr = compute_eer(yte, scores)
print(f"Balanced accuracy : {balanced_accuracy_score(yte, preds):.4f}")
print(f"ROC-AUC           : {roc_auc_score(yte, scores):.4f}")
print(f"EER               : {eer:.4f}  (threshold={eer_thr:.3f})")
print("\nConfusion matrix [rows=true, cols=pred] (0=real,1=fake):")
print(confusion_matrix(yte, preds))
print("\n", classification_report(yte, preds, target_names=["real","fake"]))


import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(yte, scores)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(yte, scores):.3f}")
plt.plot([0,1],[0,1],"--",c="gray")
plt.scatter([eer],[1-eer], c="red", zorder=5, label=f"EER={eer:.3f}")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Deepfake Detection ROC — Whisper encoder + LogReg")
plt.legend(); plt.tight_layout(); plt.show()


# 7. Inference on a single file
import librosa
@torch.no_grad()
def predict_file(path):
    a, _ = librosa.load(path, sr=SR, mono=True)
    e = embed(a.astype(np.float32))
    p = clf.predict_proba(scaler.transform(e[None, :]))[0, 1]
    return {"p_fake": float(p), "verdict": "FAKE" if p >= 0.5 else "REAL"}

# print(predict_file("sample.wav"))

## Notes & next steps

"""
- **Baseline only.** Frozen encoder + LogReg is deliberately simple. Stronger options:
  fine-tune the encoder, or use a purpose-built anti-spoofing model (AASIST, RawNet2, wav2vec2-AASIST).
- **Codec / RMS robustness.** If you plan to test across codecs, apply the same RMS
  normalization and codec augmentation to *both* train and test to avoid distribution mismatch.
- **Cross-lingual generalization.** A detector trained only on PT-BR often fails on other
  languages — evaluate zero-shot on your other language subsets before claiming generality.
- **Report EER**, since accuracy is misleading under class imbalance.
"""