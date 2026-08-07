import modal

app = modal.App("research-work-gen")

data_volume = modal.Volume.from_name("research-work-data")
VOLUMES = {"/data": data_volume}

CETUC_DIR = "/data/cetuc_extracted"
OUT_ROOT = "/data/braziliandf_v1/fakes"

yourtts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1", "ffmpeg")
    .pip_install("coqui-tts", "transformers==4.46.2", "torch", "torchaudio", "click")
)

kokoro_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install("kokoro>=0.9", "soundfile", "torch")
)

edge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("edge-tts")
)

driver_image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")


class CetucSampler:
    @staticmethod
    def sample_tasks(per_speaker, seed=13):
        import os
        import random

        rng = random.Random(seed)
        tasks = {}
        for entry in sorted(os.listdir(CETUC_DIR)):
            full = os.path.join(CETUC_DIR, entry)
            if not os.path.isdir(full) or "_" not in entry:
                continue
            code = entry.rsplit("_", 1)[1]
            utterances = []
            for filename in os.listdir(full):
                if filename.endswith(".wav") and not filename.startswith("._"):
                    transcript = os.path.join(full, filename[:-4] + ".txt")
                    if os.path.exists(transcript):
                        utterances.append(
                            (filename[:-4], os.path.join(full, filename), transcript))
            if len(utterances) < per_speaker + 1:
                continue
            picked = rng.sample(sorted(utterances), per_speaker + 1)
            selected = []
            for utt_id, wav, transcript in picked:
                with open(transcript, encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
                if text:
                    selected.append((utt_id, wav, text))
            tasks[code] = selected
        return tasks


class ManifestWriter:
    COLUMNS = ["path", "engine", "voice", "source_speaker", "utt", "text"]

    @staticmethod
    def write(engine, rows):
        import csv
        import os

        os.makedirs(OUT_ROOT, exist_ok=True)
        path = os.path.join(OUT_ROOT, f"{engine}_manifest.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ManifestWriter.COLUMNS)
            writer.writerows(rows)
        return path


@app.function(image=yourtts_image, volumes=VOLUMES, gpu="A10G", timeout=6 * 3600)
def generate_yourtts(per_speaker: int = 20):
    import os

    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to("cuda")
    tasks = CetucSampler.sample_tasks(per_speaker)
    rows, n_failures = [], 0
    for i, (code, utterances) in enumerate(sorted(tasks.items())):
        reference_wav = utterances[0][1]
        out_dir = os.path.join(OUT_ROOT, "yourtts", code)
        os.makedirs(out_dir, exist_ok=True)
        for utt_id, _, text in utterances[1:]:
            out = os.path.join(out_dir, f"{utt_id}.wav")
            if os.path.exists(out):
                rows.append([out, "yourtts", code, code, utt_id, text])
                continue
            try:
                tts.tts_to_file(text=text, speaker_wav=reference_wav,
                                language="pt-br", file_path=out)
                rows.append([out, "yourtts", code, code, utt_id, text])
            except Exception as e:
                n_failures += 1
                print(f"FAIL {code}/{utt_id}: {str(e)[:120]}")
        if (i + 1) % 10 == 0:
            data_volume.commit()
            print(f"{i + 1}/{len(tasks)} speakers, {len(rows)} clips")
    ManifestWriter.write("yourtts", rows)
    data_volume.commit()
    print(f"yourtts: {len(rows)} clips, {n_failures} failures")
    return {"engine": "yourtts", "n": len(rows), "failures": n_failures}


@app.function(image=kokoro_image, volumes=VOLUMES, timeout=4 * 3600)
def generate_kokoro(per_speaker: int = 5):
    import os

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="p")
    voices = ["pf_dora", "pm_alex", "pm_santa"]
    tasks = CetucSampler.sample_tasks(per_speaker)
    rows, n_failures = [], 0
    for i, (code, utterances) in enumerate(sorted(tasks.items())):
        out_dir = os.path.join(OUT_ROOT, "kokoro", code)
        os.makedirs(out_dir, exist_ok=True)
        for k, (utt_id, _, text) in enumerate(utterances[1:]):
            voice = voices[k % len(voices)]
            out = os.path.join(out_dir, f"{utt_id}.wav")
            if os.path.exists(out):
                rows.append([out, "kokoro", voice, code, utt_id, text])
                continue
            try:
                chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
                audio = np.concatenate([np.asarray(c) for c in chunks])
                sf.write(out, audio, 24000)
                rows.append([out, "kokoro", voice, code, utt_id, text])
            except Exception as e:
                n_failures += 1
                print(f"FAIL {code}/{utt_id}: {str(e)[:120]}")
        if (i + 1) % 20 == 0:
            data_volume.commit()
            print(f"{i + 1}/{len(tasks)} speakers, {len(rows)} clips")
    ManifestWriter.write("kokoro", rows)
    data_volume.commit()
    print(f"kokoro: {len(rows)} clips, {n_failures} failures")
    return {"engine": "kokoro", "n": len(rows), "failures": n_failures}


@app.function(image=edge_image, volumes=VOLUMES, timeout=4 * 3600)
def generate_edge(per_speaker: int = 4):
    import asyncio
    import os
    import subprocess

    import edge_tts

    voices = ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"]
    tasks = CetucSampler.sample_tasks(per_speaker)
    rows, n_failures = [], 0

    async def synthesize(text, voice, mp3_path):
        await edge_tts.Communicate(text, voice).save(mp3_path)

    async def run_all():
        nonlocal n_failures
        semaphore = asyncio.Semaphore(4)

        async def one(code, utt_id, text, k):
            nonlocal n_failures
            voice = voices[k % len(voices)]
            out_dir = os.path.join(OUT_ROOT, "edge", code)
            os.makedirs(out_dir, exist_ok=True)
            mp3 = os.path.join(out_dir, f"{utt_id}.mp3")
            wav = os.path.join(out_dir, f"{utt_id}.wav")
            if os.path.exists(wav):
                rows.append([wav, "edge", voice, code, utt_id, text])
                return
            async with semaphore:
                for attempt in range(3):
                    try:
                        await synthesize(text, voice, mp3)
                        break
                    except Exception:
                        if attempt == 2:
                            n_failures += 1
                            return
                        await asyncio.sleep(2 * (attempt + 1))
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3,
                            "-ar", "16000", "-ac", "1", wav], check=True)
            os.remove(mp3)
            rows.append([wav, "edge", voice, code, utt_id, text])

        jobs = []
        for code, utterances in sorted(tasks.items()):
            for k, (utt_id, _, text) in enumerate(utterances[1:]):
                jobs.append(one(code, utt_id, text, k))
        await asyncio.gather(*jobs)

    asyncio.run(run_all())
    ManifestWriter.write("edge", rows)
    data_volume.commit()
    print(f"edge: {len(rows)} clips, {n_failures} failures")
    return {"engine": "edge", "n": len(rows), "failures": n_failures}


@app.function(image=driver_image, timeout=12 * 3600)
def run_generation_pipeline():
    import json

    results = []
    stages = [
        ("generate_yourtts", {"per_speaker": 20}),
        ("generate_kokoro", {"per_speaker": 5}),
        ("generate_edge", {"per_speaker": 4}),
        ("build_v1_manifest", {}),
    ]
    for function_name, kwargs in stages:
        remote_fn = modal.Function.from_name("research-work", function_name)
        print(f"=== {function_name} starting ===")
        result = remote_fn.remote(**kwargs)
        print(f"=== {function_name} DONE -> {json.dumps(result)}")
        results.append({function_name: result})
    return results


@app.function(image=driver_image, volumes=VOLUMES, timeout=3600)
def build_v1_manifest():
    import glob
    import os

    import pandas as pd

    v0 = pd.read_csv("/data/braziliandf_v0/manifest.csv")
    speaker_split = (v0[v0["source"] == "cetuc"]
                     .drop_duplicates("speaker")
                     .set_index("speaker")["split"].to_dict())

    frames = [v0.assign(engine=v0["tts_engine"])]
    for manifest_path in sorted(glob.glob(os.path.join(OUT_ROOT, "*_manifest.csv"))):
        generated = pd.read_csv(manifest_path)
        generated = generated[generated["source_speaker"].isin(speaker_split)]
        gender = generated["source_speaker"].str[0].map({"F": "female", "M": "male"})
        frames.append(pd.DataFrame({
            "path": generated["path"], "label": 1,
            "speaker": generated["source_speaker"], "gender": gender,
            "tts_engine": generated["engine"], "source": "braziliandf_gen",
            "split": generated["source_speaker"].map(speaker_split),
            "engine": generated["engine"],
        }))
    v1 = pd.concat(frames, ignore_index=True)
    os.makedirs("/data/braziliandf_v1", exist_ok=True)
    v1.to_csv("/data/braziliandf_v1/manifest.csv", index=False)
    data_volume.commit()
    print(v1.groupby(["split", "label", "engine"], dropna=False).size())
    return {"n_total": int(len(v1))}
