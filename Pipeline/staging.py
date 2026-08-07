import modal

app = modal.App("research-work")

data_volume = modal.Volume.from_name("research-work-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name("research-work-outputs", create_if_missing=True)

VOLUMES = {"/data": data_volume, "/outputs": outputs_volume}
HF_SECRET = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "unar")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "datasets[audio]",
        "pandas",
        "pyarrow",
        "soundfile",
        "tqdm",
        "torch",
        "torchcodec",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


class VolumePaths:
    CETUC = "/data/cetuc"
    CETUC_EXTRACTED = "/data/cetuc_extracted"
    CORAA_MUPE = "/data/coraa_mupe"
    BRSPEECH = "/data/brspeech_df"
    LRLSPOOF_META = "/data/lrlspoof/meta"
    LRLSPOOF_PT = "/data/lrlspoof/pt"
    COMMON_VOICE = "/data/common_voice_pt"
    FAKE_VOICES = "/data/fake_voices"
    BENCHMARK_V0 = "/data/braziliandf_v0"


class HuggingFaceRepos:
    CETUC = "falabrasil/cetuc"
    CORAA_MUPE = "nilc-nlp/CORAA-MUPE-ASR"
    BRSPEECH = "AKCIT-Deepfake/BRSpeech-DF"
    LRLSPOOF = "lab260/LRLspoof"
    FAKE_VOICES = "unfake/fake_voices"
    COMMON_VOICE = "mozilla-foundation/common_voice_17_0"


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=4 * 3600)
def download_cetuc():
    from huggingface_hub import snapshot_download

    snapshot_download(
        HuggingFaceRepos.CETUC,
        repo_type="dataset",
        local_dir=VolumePaths.CETUC,
    )
    data_volume.commit()
    print(f"CETUC staged at {VolumePaths.CETUC}")


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=8 * 3600)
def download_coraa_mupe(metadata_only: bool = True):
    from huggingface_hub import snapshot_download

    patterns = ["*.csv", "*.tsv", "*.parquet", "*.json", "README.md"] if metadata_only else None
    snapshot_download(
        HuggingFaceRepos.CORAA_MUPE,
        repo_type="dataset",
        local_dir=VolumePaths.CORAA_MUPE,
        allow_patterns=patterns,
    )
    data_volume.commit()
    print(f"CORAA-MUPE staged at {VolumePaths.CORAA_MUPE} (metadata_only={metadata_only})")


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=12 * 3600)
def download_brspeech_subset(n_per_class: int = 5000, split: str = "train", seed: int = 42):
    import csv
    import os

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    wav_dir = os.path.join(VolumePaths.BRSPEECH, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    dataset = load_dataset(HuggingFaceRepos.BRSPEECH, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def label_to_int(value):
        text = str(value).strip().lower()
        return 1 if text in ("spoof", "fake", "synthetic", "1") else 0

    counts = {0: 0, 1: 0}
    rows = []
    for row in dataset:
        label = label_to_int(row.get("label"))
        if counts[label] >= n_per_class:
            if all(v >= n_per_class for v in counts.values()):
                break
            continue
        audio = row["audio"]
        filename = f"brspeech_{split}_{sum(counts.values()):06d}.wav"
        sf.write(
            os.path.join(wav_dir, filename),
            np.asarray(audio["array"], dtype=np.float32),
            16_000,
        )
        meta = {k: v for k, v in row.items() if k != "audio"}
        meta.update({"filename": filename, "label_int": label})
        rows.append(meta)
        counts[label] += 1
        if sum(counts.values()) % 500 == 0:
            print(f"  {sum(counts.values())} clips (bonafide={counts[0]}, spoof={counts[1]})")

    fieldnames = sorted({k for r in rows for k in r})
    with open(os.path.join(VolumePaths.BRSPEECH, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    data_volume.commit()
    print(f"BRSpeech-DF subset staged: {counts} at {VolumePaths.BRSPEECH}")


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 3600)
def download_lrlspoof_meta():
    from huggingface_hub import list_repo_files, snapshot_download

    files = list_repo_files(HuggingFaceRepos.LRLSPOOF, repo_type="dataset")
    parts = [f for f in files if "tar.gz.part" in f]
    print(f"Repo has {len(files)} files, {len(parts)} tarball parts")

    snapshot_download(
        HuggingFaceRepos.LRLSPOOF,
        repo_type="dataset",
        local_dir=VolumePaths.LRLSPOOF_META,
        allow_patterns=["*.parquet", "*.json", "*.csv", "*.md", "*.txt"],
    )
    data_volume.commit()

    import glob

    import pandas as pd

    for path in glob.glob(f"{VolumePaths.LRLSPOOF_META}/**/*.parquet", recursive=True):
        df = pd.read_parquet(path)
        print(f"\n{path}: {len(df):,} rows, columns={list(df.columns)}")
        print(df.head(10).to_string())


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[HF_SECRET],
    timeout=12 * 3600,
    retries=modal.Retries(max_retries=3, initial_delay=30.0),
)
def extract_lrlspoof_pt(lang_prefix: str = "brazilian"):
    import os
    import shutil
    import subprocess
    import time

    import pandas as pd
    import requests
    from huggingface_hub import get_token, list_repo_files

    labels = pd.read_parquet(f"{VolumePaths.LRLSPOOF_META}/data/labels.parquet")
    partition = labels[labels["utterance_id"].str.startswith(f"{lang_prefix}/")]
    n_systems = partition["utterance_id"].str.split("/").str[1].nunique()
    print(f"Expecting {len(partition):,} {lang_prefix} utterances across {n_systems} TTS systems")

    parts = sorted(
        f for f in list_repo_files(HuggingFaceRepos.LRLSPOOF, repo_type="dataset")
        if "tar.gz.part" in f
    )
    print(f"Streaming {len(parts)} parts")

    out_dir = VolumePaths.LRLSPOOF_PT
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tar = subprocess.Popen(
        ["tar", "-xz", "-C", out_dir, "--strip-components=1",
         "--wildcards", f"lrl_spoof/{lang_prefix}/*"],
        stdin=subprocess.PIPE,
    )
    token = get_token()
    base_headers = {"Authorization": f"Bearer {token}"} if token else {}

    def stream_part(url):
        offset = 0
        for attempt in range(6):
            headers = dict(base_headers)
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        tar.stdin.write(chunk)
                        offset += len(chunk)
                return
            except (requests.RequestException, IOError) as e:
                print(f"    retry {attempt + 1} at offset {offset}: {type(e).__name__}")
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"Failed to stream {url} after retries")

    started = time.time()
    for i, part in enumerate(parts):
        stream_part(f"https://huggingface.co/datasets/{HuggingFaceRepos.LRLSPOOF}/resolve/main/{part}")
        print(f"  part {i + 1}/{len(parts)} streamed ({(time.time() - started) / 60:.0f} min elapsed)")
    tar.stdin.close()
    print(f"tar exit code: {tar.wait()}")

    n_files = sum(len(files) for _, _, files in os.walk(out_dir))
    print(f"Extracted {n_files:,} files to {out_dir}")
    data_volume.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=8 * 3600)
def download_common_voice(version: str = HuggingFaceRepos.COMMON_VOICE):
    from huggingface_hub import snapshot_download

    snapshot_download(
        version,
        repo_type="dataset",
        local_dir=VolumePaths.COMMON_VOICE,
        allow_patterns=["*pt*", "*.py", "README.md"],
    )
    data_volume.commit()
    print(f"Common Voice PT staged at {VolumePaths.COMMON_VOICE}")


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=6 * 3600)
def download_fake_voices():
    import os
    import subprocess
    import zipfile

    from huggingface_hub import snapshot_download

    zip_root = f"{VolumePaths.FAKE_VOICES}/zips"
    snapshot_download(
        HuggingFaceRepos.FAKE_VOICES,
        repo_type="dataset",
        local_dir=zip_root,
    )

    archive_root = os.path.join(zip_root, "falabrasil-fake-voices")
    out_root = f"{VolumePaths.FAKE_VOICES}/wavs"
    os.makedirs(out_root, exist_ok=True)
    archives = sorted(
        f for f in os.listdir(archive_root)
        if f.endswith(".zip") or f.endswith(".rar")
    )
    for i, archive in enumerate(archives):
        speaker = archive.replace("_Fake.zip", "").replace("_Fake.rar", "")
        dest = os.path.join(out_root, speaker)
        if not os.path.exists(dest):
            src = os.path.join(archive_root, archive)
            if archive.endswith(".zip"):
                with zipfile.ZipFile(src) as zf:
                    zf.extractall(dest)
            else:
                subprocess.run(["unar", "-quiet", "-o", dest, src], check=False)
                n_recovered = sum(len(files) for _, _, files in os.walk(dest))
                print(f"  {speaker}: {n_recovered} files recovered from RAR")
        if (i + 1) % 20 == 0:
            print(f"  extracted {i + 1}/{len(archives)}")
    n_files = sum(len(files) for _, _, files in os.walk(out_root))
    print(f"Fake-Voices-BR: {len(archives)} speakers, {n_files:,} files at {out_root}")
    data_volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=4 * 3600)
def prepare_cetuc():
    import os
    import tarfile

    src = f"{VolumePaths.CETUC}/data"
    out_root = VolumePaths.CETUC_EXTRACTED
    n_shards = 0
    for split in ("train", "dev", "test"):
        split_dir = os.path.join(src, split)
        if not os.path.isdir(split_dir):
            continue
        for speaker in sorted(os.listdir(split_dir)):
            speaker_dir = os.path.join(split_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            dest = os.path.join(out_root, speaker)
            if os.path.exists(dest):
                continue
            os.makedirs(dest, exist_ok=True)
            for shard in os.listdir(speaker_dir):
                if shard.endswith(".tar.gz") or shard.endswith(".tar"):
                    with tarfile.open(os.path.join(speaker_dir, shard)) as tf:
                        tf.extractall(dest)
                    n_shards += 1
            if n_shards % 20 == 0:
                print(f"  {n_shards} shards extracted")
    n_files = sum(len(files) for _, _, files in os.walk(out_root))
    print(f"CETUC extracted: {n_files:,} files at {out_root}")
    data_volume.commit()


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def build_manifest(max_per_speaker: int = 200, seed: int = 42):
    import csv
    import os
    import random
    import re
    from collections import Counter

    real_root = VolumePaths.CETUC_EXTRACTED
    fake_root = f"{VolumePaths.FAKE_VOICES}/wavs"

    def speaker_code(name):
        match = re.search(r"_([FM]\d+)", name)
        return match.group(1) if match else name

    rng = random.Random(seed)
    rows = []
    speakers = set()
    for root, label, engine in ((real_root, 0, "none"), (fake_root, 1, "xtts_v2")):
        for speaker_dir in sorted(os.listdir(root)):
            full = os.path.join(root, speaker_dir)
            if not os.path.isdir(full):
                continue
            code = speaker_code(speaker_dir)
            speakers.add(code)
            wavs = []
            for dirpath, _, files in os.walk(full):
                wavs.extend(os.path.join(dirpath, f) for f in files if f.endswith(".wav"))
            rng.shuffle(wavs)
            gender = "female" if code.startswith("F") else "male"
            for wav in wavs[:max_per_speaker]:
                rows.append({
                    "path": wav, "label": label, "speaker": code,
                    "gender": gender, "tts_engine": engine,
                    "source": "cetuc" if label == 0 else "fake_voices_br",
                })

    speaker_list = sorted(speakers)
    rng.shuffle(speaker_list)
    n = len(speaker_list)
    split_of = {
        s: "train" if i < 0.8 * n else ("dev" if i < 0.9 * n else "test")
        for i, s in enumerate(speaker_list)
    }
    for row in rows:
        row["split"] = split_of[row["speaker"]]

    os.makedirs(VolumePaths.BENCHMARK_V0, exist_ok=True)
    with open(os.path.join(VolumePaths.BENCHMARK_V0, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    data_volume.commit()

    counts = Counter((r["split"], r["label"]) for r in rows)
    print(f"{len(rows):,} clips, {n} speakers")
    for key in sorted(counts):
        print(f"  split={key[0]} label={key[1]}: {counts[key]:,}")


@app.function(image=image, volumes=VOLUMES, timeout=600)
def inspect_lrlspoof_languages():
    import pandas as pd

    labels = pd.read_parquet(f"{VolumePaths.LRLSPOOF_META}/data/labels.parquet")
    languages = labels["utterance_id"].str.split("/").str[0].value_counts()
    print(f"{len(languages)} languages")
    for name, count in languages.items():
        print(f"{count:>8,}  {name}")


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def audit_coraa_mupe():
    import glob
    import os

    import pandas as pd
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(f"{VolumePaths.CORAA_MUPE}/data/*.parquet"))
    print(f"{len(shards)} shards")

    columns = None
    frames = []
    for shard in shards:
        schema_names = pq.read_schema(shard).names
        if columns is None:
            columns = [c for c in schema_names if c not in ("audio", "wav", "waveform")]
            print("metadata columns:", columns)
        frames.append(pq.read_table(shard, columns=columns).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    print(f"total segments: {len(df):,}")

    respondents = df[df["speaker_type"] == "R"] if "speaker_type" in df.columns else df
    speaker_col = "speaker_code" if "speaker_code" in respondents.columns else "speaker_id"

    out_dir = "/outputs/coraa_mupe_audit"
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# CORAA-MUPE Coverage Audit (segments from interviewees)\n"]
    for column in ("birth_state", "speaker_gender", "education", "racial_category"):
        if column not in respondents.columns:
            continue
        segments = respondents[column].fillna("(missing)").value_counts()
        speakers = respondents.groupby(respondents[column].fillna("(missing)"))[speaker_col].nunique()
        lines.append(f"\n## {column}\n")
        lines.append(f"| {column} | segments | unique speakers |")
        lines.append("|---|---|---|")
        for key in segments.index:
            lines.append(f"| {key} | {segments[key]:,} | {speakers.get(key, 0):,} |")
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "COVERAGE.md"), "w") as f:
        f.write(report)
    respondents.drop_duplicates(subset=[speaker_col]).to_csv(
        os.path.join(out_dir, "speakers.csv"), index=False)
    outputs_volume.commit()
    print(report)


@app.function(image=image, volumes=VOLUMES, timeout=600)
def audit_data():
    import os

    print(f"{'path':<60} {'files':>8} {'GB':>8}")
    for root in sorted(os.listdir("/data")) if os.path.exists("/data") else []:
        top = os.path.join("/data", root)
        n_files, n_bytes = 0, 0
        for dirpath, _, filenames in os.walk(top):
            n_files += len(filenames)
            n_bytes += sum(
                os.path.getsize(os.path.join(dirpath, f))
                for f in filenames
                if os.path.exists(os.path.join(dirpath, f))
            )
        print(f"{top:<60} {n_files:>8,} {n_bytes / 1e9:>8.2f}")


ACTIONS = {
    "audit": lambda **_: audit_data.remote(),
    "audit-coraa-mupe": lambda **_: audit_coraa_mupe.remote(),
    "download-cetuc": lambda **_: download_cetuc.remote(),
    "download-coraa-mupe": lambda **_: download_coraa_mupe.remote(metadata_only=True),
    "download-coraa-mupe-full": lambda **_: download_coraa_mupe.remote(metadata_only=False),
    "download-brspeech": lambda n_per_class, **_: download_brspeech_subset.remote(n_per_class=n_per_class),
    "download-lrlspoof-meta": lambda **_: download_lrlspoof_meta.remote(),
    "extract-lrlspoof-pt": lambda **_: extract_lrlspoof_pt.remote(),
    "inspect-lrlspoof-languages": lambda **_: inspect_lrlspoof_languages.remote(),
    "download-fake-voices": lambda **_: download_fake_voices.remote(),
    "prepare-cetuc": lambda **_: prepare_cetuc.remote(),
    "build-manifest": lambda **_: build_manifest.remote(),
    "download-common-voice": lambda **_: download_common_voice.remote(),
}


@app.local_entrypoint()
def main(action: str = "audit", n_per_class: int = 5000):
    if action not in ACTIONS:
        raise SystemExit(f"Unknown action: {action}. Choose from: {', '.join(sorted(ACTIONS))}")
    ACTIONS[action](n_per_class=n_per_class)
