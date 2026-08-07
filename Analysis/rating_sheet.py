import csv
import json
import os
import random
import shutil

PILOT_DIR = "datasets/portuguese/pilot_tts"
CLIPS_DIR = "datasets/portuguese/common_voice/clips"
OUT_DIR = "experiments/native_speaker_eval"
N_REAL_ANCHORS = 25
SEED = 42

INSTRUCTIONS = """# BrazilianDF - Native-Speaker Listening Test (Experiment 6)

Obrigado por participar! / Thank you for participating!

You will listen to short Portuguese audio clips and rate each one in
`rating_sheet.csv`. Some clips are real human recordings and some are
synthetic (computer-generated) - you will NOT be told which is which.
Please rate every clip on its own merits.

## How to rate each clip

1. **naturalness_1to5** - How natural does the voice sound?
   1 = clearly robotic/synthetic ... 5 = indistinguishable from a human
2. **accent_credibility_1to5** - Does it sound like a credible Brazilian
   Portuguese speaker?
   1 = not Brazilian at all (foreign/European/unnatural) ... 5 = fully
   credible Brazilian accent
3. **region_guess** - If you can, guess the speaker's region
   (e.g. Sao Paulo, Rio, Nordeste, Sul, Minas, Portugal, "cannot tell")
4. **intelligibility_1to5** - How easy is it to understand the words?
   1 = mostly unintelligible ... 5 = every word clear
   (The intended transcript is provided for reference - rate what you HEAR.)
5. **artifacts_yes_no** - Any audible glitches: metallic sound, buzzing,
   wrong pauses, mispronunciations, unnatural prosody? (yes/no)
6. **artifact_type** - If yes: short description (e.g. "metallic timbre",
   "wrong stress on 'frances'", "clipped ending")
7. **comments** - Anything else you noticed (optional)

## Practical notes

- Use headphones in a quiet environment.
- Listen to each clip at most 3 times before rating.
- There are no right or wrong answers for naturalness/accent - we want your
  honest perception as a native speaker.
- Expected time: ~45-60 minutes for the full sheet. Feel free to split
  across sessions; save the CSV as you go.
"""


class RatingPackageBuilder:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.audio_dir = os.path.join(OUT_DIR, "audio")

    def load_synthetic_items(self):
        with open(os.path.join(PILOT_DIR, "metadata.jsonl")) as f:
            synthetic = [json.loads(line) for line in f]
        items = [{
            "source": f"{r['tts_engine']}/{r['tts_voice']}",
            "src_path": os.path.join(PILOT_DIR, r["filename"]),
            "transcript": r["transcript"],
            "clip_id": r["clip_id"],
        } for r in synthetic]
        return synthetic, items

    def sample_real_anchors(self, synthetic):
        transcripts = {}
        for record in synthetic:
            transcripts.setdefault(record["source_clip_id"], record["transcript"])
        pool = [{
            "source": "real_common_voice",
            "src_path": os.path.join(CLIPS_DIR, path),
            "transcript": text,
            "clip_id": path,
        } for path, text in transcripts.items()
            if os.path.exists(os.path.join(CLIPS_DIR, path))]
        return self.rng.sample(pool, min(N_REAL_ANCHORS, len(pool)))

    def build(self):
        os.makedirs(self.audio_dir, exist_ok=True)
        synthetic, items = self.load_synthetic_items()
        items.extend(self.sample_real_anchors(synthetic))
        self.rng.shuffle(items)

        sheet_rows, key_rows = [], []
        for i, item in enumerate(items):
            extension = os.path.splitext(item["src_path"])[1]
            blind_name = f"eval_{i:03d}{extension}"
            shutil.copy2(item["src_path"], os.path.join(self.audio_dir, blind_name))
            sheet_rows.append({
                "item_id": f"eval_{i:03d}",
                "audio_file": f"audio/{blind_name}",
                "transcript": item["transcript"],
                "naturalness_1to5": "",
                "accent_credibility_1to5": "",
                "region_guess": "",
                "intelligibility_1to5": "",
                "artifacts_yes_no": "",
                "artifact_type": "",
                "comments": "",
            })
            key_rows.append({
                "item_id": f"eval_{i:03d}",
                "source": item["source"],
                "original_clip": item["clip_id"],
            })
        return items, sheet_rows, key_rows

    @staticmethod
    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    builder = RatingPackageBuilder()
    items, sheet_rows, key_rows = builder.build()

    RatingPackageBuilder.write_csv(os.path.join(OUT_DIR, "rating_sheet.csv"), sheet_rows)
    RatingPackageBuilder.write_csv(os.path.join(OUT_DIR, "answer_key.csv"), key_rows)
    with open(os.path.join(OUT_DIR, "INSTRUCTIONS.md"), "w") as f:
        f.write(INSTRUCTIONS)

    n_real = sum(1 for k in key_rows if k["source"] == "real_common_voice")
    print(f"Rating package written to {OUT_DIR}/")
    print(f"  items: {len(items)} total "
          f"({len(items) - n_real} synthetic + {n_real} real anchors)")
    print("  rating_sheet.csv (give to raters), answer_key.csv (KEEP PRIVATE), "
          "INSTRUCTIONS.md")


if __name__ == "__main__":
    main()
