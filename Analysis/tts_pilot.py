import argparse
import asyncio
import json
import os
import time

import pandas as pd

TSV_PATH = "datasets/portuguese/common_voice/validated.tsv"
OUT_DIR = "datasets/portuguese/pilot_tts"
SEED = 42
N_CONTRAST = 10

ENGINE_CONFIGS = [
    ("gtts", "pt_com.br", "br_generic", {"lang": "pt", "tld": "com.br"}),
    ("edge_tts", "pt-BR-FranciscaNeural", "br_generic", {}),
    ("edge_tts", "pt-BR-AntonioNeural", "br_generic", {}),
    ("coqui_vits", "tts_models/pt/cv/vits", "unknown", {}),
]

CONTRAST_CONFIGS = [
    ("gtts", "pt_pt", "european_pt", {"lang": "pt", "tld": "pt"}),
    ("edge_tts", "pt-PT-RaquelNeural", "european_pt", {}),
]

VOICE_GENDER = {
    "pt-BR-FranciscaNeural": "female",
    "pt-BR-AntonioNeural": "male",
    "pt-PT-RaquelNeural": "female",
    "pt_com.br": "unknown",
    "pt_pt": "unknown",
    "tts_models/pt/cv/vits": "unknown",
}


class SentenceSampler:
    @staticmethod
    def sample(n):
        df = pd.read_csv(TSV_PATH, sep="\t")
        df = df[df["sentence"].str.len().between(30, 120)]
        picked = df.sample(n=n, random_state=SEED)[["sentence", "path", "client_id"]]
        return picked.to_dict("records")


class Synthesizer:
    _coqui_model = None

    @staticmethod
    def gtts(text, out_path, lang, tld):
        from gtts import gTTS
        gTTS(text=text, lang=lang, tld=tld).save(out_path)

    @staticmethod
    def edge(text, out_path, voice):
        import edge_tts

        async def run():
            await edge_tts.Communicate(text, voice).save(out_path)

        asyncio.run(run())

    @classmethod
    def coqui(cls, text, out_path):
        if cls._coqui_model is None:
            from TTS.api import TTS
            cls._coqui_model = TTS("tts_models/pt/cv/vits", progress_bar=False)
        cls._coqui_model.tts_to_file(text=text, file_path=out_path)

    @classmethod
    def synthesize(cls, engine, text, out_path, voice, kwargs):
        if engine == "gtts":
            cls.gtts(text, out_path, **kwargs)
            time.sleep(0.5)
        elif engine == "edge_tts":
            cls.edge(text, out_path, voice)
        elif engine == "coqui_vits":
            cls.coqui(text, out_path)


class PilotRunner:
    def __init__(self):
        self.records = []
        self.results = []

    def run(self, configs, sentences):
        for engine, voice, region_claimed, kwargs in configs:
            ok, failed = 0, 0
            started = time.time()
            for i, sentence in enumerate(sentences):
                extension = "wav" if engine == "coqui_vits" else "mp3"
                safe_voice = voice.replace("/", "-").replace(".", "")
                filename = f"pilot_{engine}_{safe_voice}_{i:03d}.{extension}"
                out_path = os.path.join(OUT_DIR, filename)
                try:
                    Synthesizer.synthesize(
                        engine, sentence["sentence"], out_path, voice, kwargs)
                    ok += 1
                    self.records.append({
                        "clip_id": filename.rsplit(".", 1)[0],
                        "filename": filename,
                        "transcript": sentence["sentence"],
                        "label": 1,
                        "tts_engine": engine,
                        "tts_voice": voice,
                        "generation_params": json.dumps(kwargs),
                        "region": region_claimed,
                        "region_provenance": "synthetic_claimed",
                        "gender": VOICE_GENDER.get(voice, "unknown"),
                        "source_corpus": "common_voice",
                        "source_clip_id": sentence["path"],
                        "paired_speaker_id": sentence["client_id"][:16],
                    })
                except Exception as e:
                    failed += 1
                    print(f"  FAIL {engine}/{voice} #{i}: {type(e).__name__}: {e}")
            elapsed = time.time() - started
            self.results.append({
                "engine": engine, "voice": voice,
                "region_claimed": region_claimed,
                "ok": ok, "fail": failed, "seconds": round(elapsed, 1),
            })
            print(f"[{engine} / {voice}] {ok} ok, {failed} failed, {elapsed:.1f}s")

    def write_metadata(self):
        path = os.path.join(OUT_DIR, "metadata.jsonl")
        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Written: {path} ({len(self.records)} records)")

    def write_report(self, n_sentences):
        lines = ["# BrazilianDF - TTS Engine Feasibility Pilot (Experiment 5)\n"]
        lines.append(
            f"Pilot set: {n_sentences} sentences sampled from Common Voice PT "
            f"(seed {SEED}), each with a matching real recording for the "
            f"paired design. Plus a {N_CONTRAST}-sentence European Portuguese "
            "contrast set.\n")
        lines.append("## Engine support matrix\n")
        lines.append("| Engine | Voice | Accent claimed | Succeeded | Failed | Time (s) |")
        lines.append("|---|---|---|---|---|---|")
        for r in self.results:
            lines.append(f"| {r['engine']} | {r['voice']} | {r['region_claimed']} | "
                         f"{r['ok']} | {r['fail']} | {r['seconds']} |")
        lines.append("")
        lines.append("## Accent controllability findings\n")
        lines.append(
            "- **No engine tested exposes regional Brazilian accent control** "
            "(e.g. Paulista vs Nordestino vs Gaucho). Control is limited to "
            "the BR vs EU Portuguese locale distinction.")
        lines.append(
            "- gTTS: single PT voice per tld; `tld=com.br` yields PT-BR, "
            "`tld=pt` yields EU PT. No voice or accent parameters.")
        lines.append(
            "- edge-tts: 3 PT-BR neural voices (2F/1M) + 2 PT-PT voices. "
            "Voice-level (speaker) variety but no regional accent control.")
        lines.append(
            "- Coqui VITS (pt/cv/vits): single voice trained on Common Voice "
            "PT; accent is whatever dominates that corpus (not controllable).")
        lines.append("")
        lines.append("## Implications for benchmark design (Experiment 7)\n")
        lines.append(
            "- Regional accent fairness must come from **real speech accents "
            "paired with synthetic fakes of the same transcripts**, not from "
            "controlled synthetic accents - the 'real-only accents + synthetic "
            "paired fakes' design.")
        lines.append(
            "- To test controllable/cloned accents, voice-cloning engines "
            "(XTTS-v2, F5-TTS) with regional reference audio are the next "
            "candidates; they need reference clips per region from the real "
            "corpus and native-speaker validation (Experiment 6).")
        lines.append(
            "- Native-speaker quality check (Experiment 6) should rate the "
            "pilot clips for naturalness, accent credibility, intelligibility, "
            "and artifacts.")
        lines.append("")
        path = os.path.join(OUT_DIR, "PILOT_REPORT.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nWritten: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sentences", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sentences = SentenceSampler.sample(args.n_sentences)

    runner = PilotRunner()
    runner.run(ENGINE_CONFIGS, sentences)
    runner.run(CONTRAST_CONFIGS, sentences[:N_CONTRAST])
    runner.write_metadata()
    runner.write_report(args.n_sentences)


if __name__ == "__main__":
    main()
