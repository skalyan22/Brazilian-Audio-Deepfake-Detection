import modal

app = modal.App("research-work")

data_volume = modal.Volume.from_name("research-work-data")
outputs_volume = modal.Volume.from_name("research-work-outputs")

VOLUMES = {"/data": data_volume, "/outputs": outputs_volume}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "scikit-learn",
        "pandas",
        "numpy",
        "soundfile",
        "tqdm",
        "pyarrow",
        "peft",
    )
)


class AudioConfig:
    SAMPLE_RATE = 16_000
    CROP_SAMPLES = 4 * 16_000


class BenchmarkPaths:
    MANIFEST_V0 = "/data/braziliandf_v0/manifest.csv"
    MANIFEST_V1 = "/data/braziliandf_v1/manifest.csv"
    BRSPEECH_META = "/data/brspeech_df/metadata.csv"
    BRSPEECH_WAVS = "/data/brspeech_df/wavs"
    LRLSPOOF_PT = "/data/lrlspoof/pt"
    CORAA_MUPE_SHARDS = "/data/coraa_mupe/data/*.parquet"
    OUTPUTS_V0 = "/outputs/braziliandf_v0"
    OUTPUTS_V1 = "/outputs/braziliandf_v1"


REGION_BY_STATE = {
    "Acre": "North", "Amapá": "North", "Amazonas": "North", "Pará": "North",
    "Rondônia": "North", "Roraima": "North", "Tocantins": "North",
    "Alagoas": "Northeast", "Bahia": "Northeast", "Ceará": "Northeast",
    "Maranhão": "Northeast", "Paraíba": "Northeast", "Pernambuco": "Northeast",
    "Piauí": "Northeast", "Rio Grande do Norte": "Northeast",
    "Sergipe": "Northeast",
    "Distrito Federal": "Central-West", "Goiás": "Central-West",
    "Mato Grosso": "Central-West", "Mato Grosso do Sul": "Central-West",
    "Espírito Santo": "Southeast", "Minas Gerais": "Southeast",
    "Rio de Janeiro": "Southeast", "São Paulo": "Southeast",
    "Paraná": "South", "Rio Grande do Sul": "South",
    "Santa Catarina": "South",
}

CODEC_PROFILES = {
    "mp3_64k": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "opus_24k": ["-c:a", "libopus", "-b:a", "24k", "-f", "ogg"],
    "phone_8k": ["-ar", "8000", "-af", "highpass=f=300,lowpass=f=3400",
                 "-c:a", "pcm_s16le", "-f", "wav"],
}


def load_waveform(path, train):
    import soundfile as sf
    import torch
    import torchaudio

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != AudioConfig.SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, AudioConfig.SAMPLE_RATE)
    wav = wav.squeeze(0)
    crop = AudioConfig.CROP_SAMPLES
    if len(wav) >= crop:
        if train:
            start = torch.randint(0, len(wav) - crop + 1, (1,)).item()
        else:
            start = (len(wav) - crop) // 2
        wav = wav[start:start + crop]
    else:
        wav = torch.nn.functional.pad(wav, (0, crop - len(wav)))
    return wav


class ManifestDataset:
    def __init__(self, df, train):
        self.rows = df.to_dict("records")
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return load_waveform(row["path"], self.train), int(row["label"])


class PathLabelDataset:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, label = self.pairs[i]
        return load_waveform(path, train=False), label


class WaveBytesDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import io

        import soundfile as sf
        import torch
        import torchaudio

        raw, label = self.items[i]
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != AudioConfig.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, AudioConfig.SAMPLE_RATE)
        wav = wav.squeeze(0)
        crop = AudioConfig.CROP_SAMPLES
        if len(wav) >= crop:
            start = (len(wav) - crop) // 2
            wav = wav[start:start + crop]
        else:
            wav = torch.nn.functional.pad(wav, (0, crop - len(wav)))
        return wav, label


class MelSpectrogramCNN:
    def __init__(self, device):
        import torch.nn as nn
        import torchaudio
        from torchvision.models import resnet18

        network = resnet18(weights=None)
        network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        network.fc = nn.Linear(network.fc.in_features, 2)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=AudioConfig.SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=64
        ).to(device)
        self.to_db = torchaudio.transforms.AmplitudeToDB().to(device)
        self.net = network.to(device)
        self.device = device
        self.params = self.net.parameters()

    def __call__(self, wav):
        x = self.to_db(self.mel(wav)).unsqueeze(1)
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class FrozenWav2Vec2:
    MODEL_ID = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, device):
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained(self.MODEL_ID).to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        ).to(device)
        self.device = device
        self.params = self.head.parameters()
        self._torch = torch

    def __call__(self, wav):
        with self._torch.no_grad():
            hidden = self.encoder(wav).last_hidden_state
        return self.head(hidden.mean(1))

    def state_dict(self):
        return self.head.state_dict()

    def load_state_dict(self, sd):
        self.head.load_state_dict(sd)

    def train(self):
        self.head.train()

    def eval(self):
        self.head.eval()


BASELINES = {"cnn": MelSpectrogramCNN, "w2v": FrozenWav2Vec2}


def build_baseline(name, device):
    return BASELINES[name](device)


def score_loader(model, loader, device):
    import numpy as np
    import torch

    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for wav, y in loader:
            logits = model(wav.to(device))
            scores.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            labels.extend(y.numpy())
    return np.array(labels), np.array(scores)


def equal_error_rate(labels, scores):
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def group_rates(df, group_col, threshold, min_n=25):
    out = {}
    for group, sub in df.groupby(group_col, observed=True):
        real = sub[sub["label"] == 0]
        fake = sub[sub["label"] == 1]
        if len(sub) < min_n:
            continue
        entry = {"n_real": int(len(real)), "n_fake": int(len(fake))}
        if "speaker" in sub.columns:
            entry["n_speakers"] = int(sub["speaker"].nunique())
        if len(real):
            entry["fpr"] = float((real["score"] > threshold).mean())
        if len(fake):
            entry["fnr"] = float((fake["score"] <= threshold).mean())
        out[str(group)] = entry
    return out


def load_checkpointed_baseline(baseline, device):
    import os

    import torch

    out_dir = f"{BenchmarkPaths.OUTPUTS_V0}/{baseline}"
    checkpoint = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model = build_baseline(baseline, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, float(checkpoint["dev_threshold"]), out_dir


class Trainer:
    def __init__(self, model, device, lr, weight_decay=1e-4):
        import torch

        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.params, lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._torch = torch

    def train_epoch(self, loader, desc):
        from tqdm import tqdm

        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for inputs, y in tqdm(loader, desc=desc):
            inputs, y = inputs.to(self.device), y.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        return loss_sum / total, correct / total

    def fit(self, train_loader, dev_loader, epochs, checkpoint_path=None,
            tag="", commit_on_save=True, verbose_train_metrics=False):
        best_eer, best_threshold = 1.0, 0.5
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, f"{tag}epoch {epoch + 1}/{epochs}")
            labels, scores = score_loader(self.model, dev_loader, self.device)
            dev_eer, dev_threshold = equal_error_rate(labels, scores)
            if verbose_train_metrics:
                print(f"{tag}epoch {epoch + 1}: train_loss={train_loss:.4f} "
                      f"train_acc={train_acc:.4f} dev_eer={dev_eer * 100:.2f}%")
            else:
                print(f"{tag}epoch {epoch + 1}: dev_eer={dev_eer * 100:.2f}%")
            if dev_eer < best_eer:
                best_eer, best_threshold = dev_eer, dev_threshold
                if checkpoint_path:
                    self._torch.save(
                        {"state_dict": self.model.state_dict(),
                         "dev_eer": dev_eer, "dev_threshold": dev_threshold},
                        checkpoint_path)
                    if commit_on_save:
                        outputs_volume.commit()
        return best_eer, best_threshold

    def load_checkpoint(self, checkpoint_path):
        checkpoint = self._torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint


def build_ood_loaders(batch_size):
    import glob
    import os

    import pandas as pd
    from torch.utils.data import DataLoader

    brspeech = pd.read_csv(BenchmarkPaths.BRSPEECH_META)
    brspeech_pairs = [
        (os.path.join(BenchmarkPaths.BRSPEECH_WAVS, r.filename), int(r.label_int))
        for r in brspeech.itertuples()
    ]
    brspeech_loader = DataLoader(
        PathLabelDataset(brspeech_pairs), batch_size=batch_size, num_workers=8)

    lrlspoof_wavs = glob.glob(f"{BenchmarkPaths.LRLSPOOF_PT}/**/*.wav", recursive=True)
    lrlspoof_loader = DataLoader(
        PathLabelDataset([(p, 1) for p in lrlspoof_wavs]),
        batch_size=batch_size, num_workers=8)
    return brspeech_loader, lrlspoof_loader


def evaluate_generalization(model, threshold, brspeech_loader, lrlspoof_loader, device):
    labels, scores = score_loader(model, brspeech_loader, device)
    brspeech_eer, _ = equal_error_rate(labels, scores)
    results = {
        "brspeech_eer": brspeech_eer,
        "brspeech_srr_at_thr": float((scores[labels == 1] > threshold).mean()),
    }
    _, scores = score_loader(model, lrlspoof_loader, device)
    results["lrlspoof_srr_at_thr"] = float((scores > threshold).mean())
    return results


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def train_baseline(baseline: str = "cnn", epochs: int = 3, batch_size: int = 64,
                   lr: float = 1e-4, eval_only: bool = False):
    import json
    import os

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    device = "cuda"
    df = pd.read_csv(BenchmarkPaths.MANIFEST_V0)
    print(df.groupby(["split", "label"]).size())

    if baseline == "w2v":
        batch_size = min(batch_size, 32)

    loaders = {
        split: DataLoader(
            ManifestDataset(df[df["split"] == split], train=(split == "train")),
            batch_size=batch_size, shuffle=(split == "train"),
            num_workers=8, pin_memory=True, drop_last=(split == "train"),
        )
        for split in ("train", "dev", "test")
    }

    model = build_baseline(baseline, device)
    out_dir = f"{BenchmarkPaths.OUTPUTS_V0}/{baseline}"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best.pt")

    trainer = Trainer(model, device, lr)
    if not eval_only:
        trainer.fit(loaders["train"], loaders["dev"], epochs,
                    checkpoint_path=checkpoint_path, verbose_train_metrics=True)

    checkpoint = trainer.load_checkpoint(checkpoint_path)
    labels, scores = score_loader(model, loaders["test"], device)
    test_eer, _ = equal_error_rate(labels, scores)
    results = {
        "baseline": baseline,
        "dev_eer": checkpoint["dev_eer"],
        "dev_threshold": checkpoint["dev_threshold"],
        "test_eer": test_eer,
        "test_auc": float(roc_auc_score(labels, scores)),
        "test_acc": float(((scores > 0.5).astype(int) == labels).mean()),
        "n_test": int(len(labels)),
    }
    with open(os.path.join(out_dir, "results_indomain.json"), "w") as f:
        json.dump(results, f, indent=2)
    outputs_volume.commit()
    print(json.dumps(results, indent=2))
    return results


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=4 * 3600)
def eval_ood(baseline: str = "cnn", batch_size: int = 64):
    import glob
    import json
    import os

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    device = "cuda"
    model, threshold, out_dir = load_checkpointed_baseline(baseline, device)
    results = {"baseline": baseline, "dev_threshold": threshold}

    meta = pd.read_csv(BenchmarkPaths.BRSPEECH_META)
    pairs = [
        (os.path.join(BenchmarkPaths.BRSPEECH_WAVS, r.filename), int(r.label_int))
        for r in meta.itertuples()
    ]
    loader = DataLoader(PathLabelDataset(pairs), batch_size=batch_size, num_workers=8)
    labels, scores = score_loader(model, loader, device)
    ood_eer, _ = equal_error_rate(labels, scores)
    results["brspeech_df"] = {
        "eer": ood_eer,
        "auc": float(roc_auc_score(labels, scores)),
        "srr_at_dev_thr": float((scores[labels == 1] > threshold).mean()),
        "bona_fpr_at_dev_thr": float((scores[labels == 0] > threshold).mean()),
        "n": int(len(labels)),
    }

    lrlspoof_wavs = glob.glob(f"{BenchmarkPaths.LRLSPOOF_PT}/**/*.wav", recursive=True)
    if lrlspoof_wavs:
        loader = DataLoader(PathLabelDataset([(p, 1) for p in lrlspoof_wavs]),
                            batch_size=batch_size, num_workers=8)
        _, scores = score_loader(model, loader, device)
        results["lrlspoof_pt"] = {
            "srr_at_dev_thr": float((scores > threshold).mean()),
            "n": int(len(scores)),
        }
    else:
        results["lrlspoof_pt"] = "not staged yet"

    with open(os.path.join(out_dir, "results_ood.json"), "w") as f:
        json.dump(results, f, indent=2)
    outputs_volume.commit()
    print(json.dumps(results, indent=2))
    return results


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=4 * 3600)
def eval_fairness(baseline: str = "cnn", batch_size: int = 64,
                  mupe_per_state: int = 500, mupe_per_speaker: int = 40):
    import glob
    import json
    import os

    import pandas as pd
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader

    device = "cuda"
    model, threshold, out_dir = load_checkpointed_baseline(baseline, device)
    results = {"baseline": baseline, "dev_threshold": threshold}

    df = pd.read_csv(BenchmarkPaths.MANIFEST_V0)
    test = df[df["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(ManifestDataset(test, train=False),
                        batch_size=batch_size, num_workers=8)
    _, scores = score_loader(model, loader, device)
    test = test.assign(score=scores)
    results["test_by_gender"] = group_rates(test, "gender", threshold)

    shards = sorted(glob.glob(BenchmarkPaths.CORAA_MUPE_SHARDS))
    rows, items = [], []
    per_state, per_speaker = {}, {}
    for shard in shards:
        table = pq.ParquetFile(shard).read(
            columns=["audio", "birth_state", "speaker_gender", "age",
                     "speaker_code", "duration"])
        for i in range(table.num_rows):
            state = table["birth_state"][i].as_py()
            region = REGION_BY_STATE.get(state)
            if region is None:
                continue
            if per_state.get(state, 0) >= mupe_per_state:
                continue
            speaker = table["speaker_code"][i].as_py()
            if mupe_per_speaker and per_speaker.get(speaker, 0) >= mupe_per_speaker:
                continue
            duration = table["duration"][i].as_py() or 0
            if duration < 1.0:
                continue
            per_state[state] = per_state.get(state, 0) + 1
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
            audio = table["audio"][i].as_py()
            items.append((audio["bytes"], 0))
            rows.append({
                "state": state, "region": region,
                "gender": table["speaker_gender"][i].as_py(),
                "age": table["age"][i].as_py(),
                "speaker": speaker,
            })
        if not mupe_per_speaker \
                and all(v >= mupe_per_state for v in per_state.values()) \
                and len(per_state) >= 20:
            break
    print(f"CORAA-MUPE sample: {len(items)} clips, {len(per_state)} states, "
          f"{len(per_speaker)} speakers")

    loader = DataLoader(WaveBytesDataset(items), batch_size=batch_size, num_workers=8)
    _, scores = score_loader(model, loader, device)
    mupe = pd.DataFrame(rows).assign(score=scores, label=0)
    mupe["age_band"] = pd.cut(
        mupe["age"].astype(float),
        bins=[0, 30, 45, 60, 200],
        labels=["<=30", "31-45", "46-60", ">60"])
    results["mupe_overall_fpr"] = float((scores > threshold).mean())
    results["mupe_n"] = int(len(scores))
    results["mupe_by_region"] = group_rates(mupe, "region", threshold)
    results["mupe_by_gender"] = group_rates(mupe, "gender", threshold)
    results["mupe_by_age_band"] = group_rates(mupe, "age_band", threshold)

    with open(os.path.join(out_dir, "results_fairness.json"), "w") as f:
        json.dump(results, f, indent=2)
    outputs_volume.commit()
    print(json.dumps(results, indent=2))
    return results


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def eval_codec(baseline: str = "cnn", batch_size: int = 64):
    import json
    import os
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    device = "cuda"
    model, threshold, out_dir = load_checkpointed_baseline(baseline, device)
    results = {"baseline": baseline, "dev_threshold": threshold}

    df = pd.read_csv(BenchmarkPaths.MANIFEST_V0)
    test = df[df["split"] == "test"].reset_index(drop=True)

    def transcode(job):
        j, path, label, args, tmp = job
        encoded = os.path.join(tmp, f"{j}.bin")
        decoded = os.path.join(tmp, f"{j}.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        *args, encoded], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", encoded,
                        "-ar", str(AudioConfig.SAMPLE_RATE), "-ac", "1", decoded],
                       check=True)
        os.remove(encoded)
        return decoded, label

    for name, args in CODEC_PROFILES.items():
        tmp = tempfile.mkdtemp(prefix=f"codec_{name}_")
        jobs = [(j, r["path"], int(r["label"]), args, tmp)
                for j, r in test.iterrows()]
        with ThreadPoolExecutor(16) as executor:
            pairs = list(executor.map(transcode, jobs))

        loader = DataLoader(PathLabelDataset(pairs), batch_size=batch_size,
                            num_workers=8)
        labels, scores = score_loader(model, loader, device)
        eer, _ = equal_error_rate(labels, scores)
        results[name] = {
            "eer": eer,
            "auc": float(roc_auc_score(labels, scores)),
            "acc_at_dev_thr": float(((scores > threshold).astype(int) == labels).mean()),
            "fpr_at_dev_thr": float((scores[labels == 0] > threshold).mean()),
            "fnr_at_dev_thr": float((scores[labels == 1] <= threshold).mean()),
            "n": int(len(labels)),
        }
        print(name, json.dumps(results[name]))
        for path, _ in pairs:
            os.remove(path)

    with open(os.path.join(out_dir, "results_codec.json"), "w") as f:
        json.dump(results, f, indent=2)
    outputs_volume.commit()
    print(json.dumps(results, indent=2))
    return results


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=2 * 3600)
def eval_calibration(baseline: str = "cnn", batch_size: int = 64, n_bins: int = 10):
    import json
    import os

    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    device = "cuda"
    model, threshold, out_dir = load_checkpointed_baseline(baseline, device)

    df = pd.read_csv(BenchmarkPaths.MANIFEST_V0)
    test = df[df["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(ManifestDataset(test, train=False),
                        batch_size=batch_size, num_workers=8)
    labels, scores = score_loader(model, loader, device)

    confidence = np.maximum(scores, 1 - scores)
    predictions = (scores > 0.5).astype(int)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece, reliability = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.sum() == 0:
            continue
        accuracy = float(correct[mask].mean())
        avg_confidence = float(confidence[mask].mean())
        ece += mask.mean() * abs(accuracy - avg_confidence)
        reliability.append({"bin": [float(lo), float(hi)],
                            "n": int(mask.sum()), "acc": accuracy,
                            "conf": avg_confidence})
    results = {
        "baseline": baseline,
        "ece": float(ece),
        "brier": float(np.mean((scores - labels) ** 2)),
        "mean_conf": float(confidence.mean()),
        "acc": float(correct.mean()),
        "n": int(len(labels)),
        "reliability": reliability,
    }
    np.savez(os.path.join(out_dir, "test_scores.npz"), ys=labels, ps=scores)
    with open(os.path.join(out_dir, "results_calibration.json"), "w") as f:
        json.dump(results, f, indent=2)
    outputs_volume.commit()
    print(json.dumps(results, indent=2))
    return results


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=4 * 3600)
def export_scores(batch_size: int = 64):
    import os
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from torch.utils.data import DataLoader

    device = "cuda"

    df = pd.read_csv(BenchmarkPaths.MANIFEST_V0)
    test = df[df["split"] == "test"].reset_index(drop=True)

    meta = pd.read_csv(BenchmarkPaths.BRSPEECH_META)
    meta["path"] = meta["filename"].map(
        lambda f: os.path.join(BenchmarkPaths.BRSPEECH_WAVS, f))

    def audio_properties(path):
        try:
            info = sf.info(path)
            data, _ = sf.read(path, dtype="float32", always_2d=True)
            return info.frames / info.samplerate, float(np.sqrt((data ** 2).mean()))
        except Exception:
            return None, None

    for frame in (test, meta):
        with ThreadPoolExecutor(32) as executor:
            props = list(executor.map(audio_properties, frame["path"]))
        frame["duration"] = [d for d, _ in props]
        frame["rms"] = [r for _, r in props]

    for baseline in ("cnn", "w2v"):
        model, threshold, out_dir = load_checkpointed_baseline(baseline, device)

        loader = DataLoader(ManifestDataset(test, train=False),
                            batch_size=batch_size, num_workers=8)
        _, scores = score_loader(model, loader, device)
        test.assign(score=scores).to_csv(
            os.path.join(out_dir, "scores_test.csv"), index=False)

        pairs = [(p, int(l)) for p, l in zip(meta["path"], meta["label_int"])]
        loader = DataLoader(PathLabelDataset(pairs), batch_size=batch_size,
                            num_workers=8)
        _, scores = score_loader(model, loader, device)
        meta.assign(score=scores).to_csv(
            os.path.join(out_dir, "scores_brspeech.csv"), index=False)
        print(f"{baseline}: exported (thr={threshold:.4f})")

    outputs_volume.commit()
    return "done"


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=8 * 3600)
def train_engine_disjoint(baseline: str = "cnn", epochs: int = 3,
                          batch_size: int = 64, lr: float = 1e-4):
    import json
    import os

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    device = "cuda"
    df = pd.read_csv(BenchmarkPaths.MANIFEST_V1)
    if baseline == "w2v":
        batch_size = min(batch_size, 32)

    test = df[df["split"] == "test"].reset_index(drop=True)
    test_loader = DataLoader(ManifestDataset(test, train=False),
                             batch_size=batch_size, num_workers=8)
    brspeech_loader, lrlspoof_loader = build_ood_loaders(batch_size)

    configs = [
        ("holdout_kokoro", ["xtts_v2", "yourtts", "edge"]),
        ("all_engines", ["xtts_v2", "yourtts", "edge", "kokoro"]),
    ]
    all_results = {}
    for name, engines in configs:
        keep = (df["label"] == 0) | df["engine"].isin(engines)
        subset = df[keep]
        train_df = subset[subset["split"] == "train"]
        dev_df = subset[subset["split"] == "dev"]
        print(f"--- {name}: train {len(train_df)}, dev {len(dev_df)} ---")

        train_loader = DataLoader(ManifestDataset(train_df, train=True),
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=True)
        dev_loader = DataLoader(ManifestDataset(dev_df, train=False),
                                batch_size=batch_size, num_workers=8)

        model = build_baseline(baseline, device)
        out_dir = f"{BenchmarkPaths.OUTPUTS_V1}/{baseline}_{name}"
        os.makedirs(out_dir, exist_ok=True)
        checkpoint_path = os.path.join(out_dir, "best.pt")

        trainer = Trainer(model, device, lr)
        trainer.fit(train_loader, dev_loader, epochs,
                    checkpoint_path=checkpoint_path, tag=f"{name} ")
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        threshold = checkpoint["dev_threshold"]

        labels, scores = score_loader(model, test_loader, device)
        test_eer, _ = equal_error_rate(labels, scores)
        result = {
            "train_engines": engines,
            "dev_eer": checkpoint["dev_eer"],
            "dev_threshold": float(threshold),
            "test_eer": test_eer,
            "test_auc": float(roc_auc_score(labels, scores)),
        }
        test_scores = test.assign(score=scores)
        for engine, group in test_scores[test_scores["label"] == 1].groupby("engine"):
            result[f"test_fnr_{engine}"] = float((group["score"] <= threshold).mean())
        result["test_fpr_real"] = float(
            (test_scores[test_scores["label"] == 0]["score"] > threshold).mean())

        result.update(evaluate_generalization(
            model, threshold, brspeech_loader, lrlspoof_loader, device))

        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        outputs_volume.commit()
        print(name, json.dumps(result, indent=2))
        all_results[name] = result

    return all_results


def train_and_evaluate_cnn(train_df, dev_df, test_loader, brspeech_loader,
                           lrlspoof_loader, out_dir, epochs, batch_size, lr, device):
    import json
    import os

    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    train_loader = DataLoader(ManifestDataset(train_df, train=True),
                              batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(ManifestDataset(dev_df, train=False),
                            batch_size=batch_size, num_workers=8)

    model = build_baseline("cnn", device)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "best.pt")

    trainer = Trainer(model, device, lr)
    trainer.fit(train_loader, dev_loader, epochs,
                checkpoint_path=checkpoint_path, commit_on_save=False)
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    threshold = checkpoint["dev_threshold"]

    labels, scores = score_loader(model, test_loader, device)
    test_eer, _ = equal_error_rate(labels, scores)
    result = {"dev_eer": checkpoint["dev_eer"], "dev_threshold": float(threshold),
              "test_eer": test_eer, "test_auc": float(roc_auc_score(labels, scores)),
              "n_train": int(len(train_df))}
    result.update(evaluate_generalization(
        model, threshold, brspeech_loader, lrlspoof_loader, device))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=10 * 3600)
def train_scaling(epochs: int = 3, batch_size: int = 64, lr: float = 1e-4):
    import json

    import pandas as pd
    from torch.utils.data import DataLoader

    device = "cuda"
    df = pd.read_csv(BenchmarkPaths.MANIFEST_V1)
    test = df[df["split"] == "test"].reset_index(drop=True)
    test_loader = DataLoader(ManifestDataset(test, train=False),
                             batch_size=batch_size, num_workers=8)
    brspeech_loader, lrlspoof_loader = build_ood_loaders(batch_size)

    train_full = df[df["split"] == "train"]
    dev_full = df[df["split"] == "dev"]
    all_results = {}

    for fraction in (0.10, 0.25, 0.50, 1.00):
        name = f"size_{int(fraction * 100)}"
        train_df = (train_full if fraction == 1.0
                    else train_full.sample(frac=fraction, random_state=7))
        result = train_and_evaluate_cnn(
            train_df, dev_full, test_loader, brspeech_loader, lrlspoof_loader,
            f"{BenchmarkPaths.OUTPUTS_V1}/scaling/{name}", epochs, batch_size,
            lr, device)
        outputs_volume.commit()
        print(name, json.dumps(result))
        all_results[name] = result

    fakes = train_full[train_full["label"] == 1]
    real = train_full[train_full["label"] == 0]
    multi_engine = fakes[fakes["engine"] != "xtts_v2"]
    budget = len(multi_engine) + len(fakes[fakes["engine"] == "xtts_v2"].sample(
        n=min(1100, (fakes["engine"] == "xtts_v2").sum()), random_state=7))
    configs = {
        "budget_1eng": fakes[fakes["engine"] == "xtts_v2"].sample(
            n=budget, random_state=7),
        "budget_4eng": pd.concat([
            multi_engine,
            fakes[fakes["engine"] == "xtts_v2"].sample(
                n=budget - len(multi_engine), random_state=7),
        ]),
    }
    for name, fake_df in configs.items():
        train_df = pd.concat([real, fake_df])
        engines = set(fake_df["engine"])
        dev_df = dev_full[(dev_full["label"] == 0)
                          | dev_full["engine"].isin(engines)]
        result = train_and_evaluate_cnn(
            train_df, dev_df, test_loader, brspeech_loader, lrlspoof_loader,
            f"{BenchmarkPaths.OUTPUTS_V1}/scaling/{name}", epochs, batch_size,
            lr, device)
        result["n_fakes"] = int(len(fake_df))
        outputs_volume.commit()
        print(name, json.dumps(result))
        all_results[name] = result

    return all_results


class WhisperClassifier:
    MODEL_ID = "openai/whisper-small"

    def __init__(self, device, mode):
        import torch
        import torch.nn as nn
        from transformers import WhisperModel

        encoder = WhisperModel.from_pretrained(self.MODEL_ID).encoder.to(device)
        hidden = encoder.config.hidden_size
        if mode == "frozen":
            for p in encoder.parameters():
                p.requires_grad = False
        elif mode == "lora":
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                target_modules=["q_proj", "v_proj"])
            encoder = get_peft_model(encoder, config)
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2)).to(device)
        self.mode = mode
        self.device = device
        self._torch = torch
        self.params = (list(self.head.parameters()) +
                       [p for p in encoder.parameters() if p.requires_grad])

    def __call__(self, features):
        if self.mode == "frozen":
            with self._torch.no_grad():
                hidden = self.encoder(features).last_hidden_state
        else:
            hidden = self.encoder(features).last_hidden_state
        return self.head(hidden.mean(1))

    def state_dict(self):
        return {"head": self.head.state_dict()}

    def train(self):
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()


class WhisperDataset:
    def __init__(self, df, train):
        self.rows = df.to_dict("records")
        self.train = train
        self._feature_extractor = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._feature_extractor is None:
            from transformers import WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                WhisperClassifier.MODEL_ID)
        row = self.rows[i]
        wav = load_waveform(row["path"], self.train)
        features = self._feature_extractor(
            wav.numpy(), sampling_rate=AudioConfig.SAMPLE_RATE,
            return_tensors="pt").input_features[0]
        return features, int(row["label"])


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=12 * 3600)
def train_whisper(epochs: int = 2, batch_size: int = 16):
    import glob
    import json
    import os

    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    device = "cuda"
    df = pd.read_csv(BenchmarkPaths.MANIFEST_V1)
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"]
    test = df[df["split"] == "test"].reset_index(drop=True)

    brspeech = pd.read_csv(BenchmarkPaths.BRSPEECH_META)
    brspeech_df = pd.DataFrame({
        "path": [f"{BenchmarkPaths.BRSPEECH_WAVS}/" + f for f in brspeech["filename"]],
        "label": brspeech["label_int"].astype(int)})
    lrlspoof_df = pd.DataFrame({
        "path": glob.glob(f"{BenchmarkPaths.LRLSPOOF_PT}/**/*.wav", recursive=True)})
    lrlspoof_df["label"] = 1

    def make_loader(frame, train=False):
        return DataLoader(WhisperDataset(frame, train=train), batch_size=batch_size,
                          shuffle=train, num_workers=8, pin_memory=True,
                          drop_last=train)

    all_results = {}
    for mode, lr in (("frozen", 1e-4), ("lora", 1e-4), ("full", 1e-5)):
        print(f"--- whisper {mode} ---")
        model = WhisperClassifier(device, mode)
        out_dir = f"{BenchmarkPaths.OUTPUTS_V1}/whisper/{mode}"
        os.makedirs(out_dir, exist_ok=True)

        trainer = Trainer(model, device, lr)
        best_eer, threshold = trainer.fit(
            make_loader(train_df, train=True), make_loader(dev_df), epochs,
            tag=f"{mode} ")

        labels, scores = score_loader(model, make_loader(test), device)
        test_eer, _ = equal_error_rate(labels, scores)
        result = {"mode": mode, "dev_eer": best_eer, "dev_threshold": float(threshold),
                  "test_eer": test_eer, "test_auc": float(roc_auc_score(labels, scores))}

        labels, scores = score_loader(model, make_loader(brspeech_df), device)
        brspeech_eer, _ = equal_error_rate(labels, scores)
        result["brspeech_eer"] = brspeech_eer
        result["brspeech_srr_at_thr"] = float((scores[labels == 1] > threshold).mean())
        _, scores = score_loader(model, make_loader(lrlspoof_df), device)
        result["lrlspoof_srr_at_thr"] = float((scores > threshold).mean())

        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        outputs_volume.commit()
        print(mode, json.dumps(result, indent=2))
        all_results[mode] = result
        del model
        torch.cuda.empty_cache()

    return all_results


@app.function(image=image, volumes={"/data": data_volume}, timeout=6 * 3600)
def build_degraded_set():
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    df = pd.read_csv(BenchmarkPaths.MANIFEST_V1)
    test = df[df["split"] == "test"].reset_index(drop=True)
    rows = []

    def transcode(job):
        j, row, name, args = job
        out_dir = f"/data/braziliandf_v1/degraded/{name}"
        os.makedirs(out_dir, exist_ok=True)
        encoded = os.path.join(out_dir, f"{j}.bin")
        decoded = os.path.join(out_dir, f"{j}.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", row["path"],
                        *args, encoded], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", encoded,
                        "-ar", str(AudioConfig.SAMPLE_RATE), "-ac", "1", decoded],
                       check=True)
        os.remove(encoded)
        return {**{k: row[k] for k in ("label", "speaker", "gender", "engine")},
                "path": decoded, "codec": name, "clean_path": row["path"]}

    for name, args in CODEC_PROFILES.items():
        jobs = [(j, row, name, args) for j, row in test.iterrows()]
        with ThreadPoolExecutor(16) as executor:
            rows.extend(executor.map(transcode, jobs))
        print(f"{name}: done ({len(test)} clips)")
        data_volume.commit()

    pd.DataFrame(rows).to_csv(
        "/data/braziliandf_v1/degraded/manifest.csv", index=False)
    data_volume.commit()
    return {"n": len(rows)}


@app.local_entrypoint()
def main(baseline: str = "cnn", epochs: int = 3, eval_only: bool = False,
         ood: bool = False):
    if ood:
        eval_ood.remote(baseline=baseline)
    else:
        train_baseline.remote(baseline=baseline, epochs=epochs, eval_only=eval_only)
