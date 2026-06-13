import json
import shutil
import time
from pathlib import Path
from collections import defaultdict

from PIL import Image
from datasets import Dataset, DatasetDict, Image as HFImage, Features, Value, Sequence
from huggingface_hub import HfApi
from tqdm import tqdm

OUTPUT_DIR = Path("data")
IMAGES_DIR = OUTPUT_DIR / "images"
RECORDS_PATH = OUTPUT_DIR / "scraped_records.json"
PARQUET_DIR = OUTPUT_DIR / "parquet_shards"
REPO_ID = "Sigurdur/19th-century-icelandic-letters"

def load_image_from_disk(lid, idx):
    path = IMAGES_DIR / str(lid) / f"{idx}.jpg"
    if path.exists():
        return Image.open(path).convert("RGB")
    return None

def make_features():
    return Features({
        "id": Value("int32"),
        "filename": Value("string"),
        "old_filename": Value("string"),
        "date": Value("string"),
        "place_farm": Value("string"),
        "place_municipality": Value("string"),
        "place_county": Value("string"),
        "note": Value("string"),
        "collection": Value("string"),
        "shelfmark": Value("string"),
        "recipient_name": Value("string"),
        "recipient_title": Value("string"),
        "image_status": Value("string"),
        "writer_name": Value("string"),
        "writer_title": Value("string"),
        "writer_gender": Value("string"),
        "writer_birth_date": Value("string"),
        "writer_death_date": Value("string"),
        "writer_birth_place_farm": Value("string"),
        "writer_birth_place_municipality": Value("string"),
        "writer_birth_place_county": Value("string"),
        "writer_origin_farm": Value("string"),
        "writer_origin_municipality": Value("string"),
        "writer_origin_county": Value("string"),
        "text_plain": Value("string"),
        "text_html": Value("string"),
        "images": Sequence(HFImage(decode=True)),
        "image_urls": Sequence(Value("string")),
    })

def split_records(records):
    writers = defaultdict(list)
    for r in records:
        w = r.get("writer_name", "unknown")
        writers[w].append(r)

    writer_items = list(writers.items())
    writer_items.sort(key=lambda x: len(x[1]), reverse=True)

    train_recs, val_recs, test_recs = [], [], []
    for writer, recs in writer_items:
        n = len(recs)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            remaining = n - n_train
            n_val = remaining // 2
            n_test = remaining - n_val
        sorted_recs = sorted(recs, key=lambda x: x["id"])
        train_recs.extend(sorted_recs[:n_train])
        val_recs.extend(sorted_recs[n_train:n_train + n_val])
        test_recs.extend(sorted_recs[n_train + n_val:])

    return train_recs, val_recs, test_recs

def record_generator(records):
    for r in records:
        images = []
        for i in range(r["image_count"]):
            img = load_image_from_disk(r["id"], i)
            if img is not None:
                images.append(img)
        yield {
            "id": r["id"],
            "filename": r.get("filename", ""),
            "old_filename": r.get("old_filename", ""),
            "date": r.get("date", ""),
            "place_farm": r.get("place_farm", ""),
            "place_municipality": r.get("place_municipality", ""),
            "place_county": r.get("place_county", ""),
            "note": r.get("note", ""),
            "collection": r.get("collection", ""),
            "shelfmark": r.get("shelfmark", ""),
            "recipient_name": r.get("recipient_name", ""),
            "recipient_title": r.get("recipient_title", ""),
            "image_status": r.get("image_status", ""),
            "writer_name": r.get("writer_name", ""),
            "writer_title": r.get("writer_title", ""),
            "writer_gender": r.get("writer_gender", ""),
            "writer_birth_date": r.get("writer_birth_date", ""),
            "writer_death_date": r.get("writer_death_date", ""),
            "writer_birth_place_farm": r.get("writer_birth_place_farm", ""),
            "writer_birth_place_municipality": r.get("writer_birth_place_municipality", ""),
            "writer_birth_place_county": r.get("writer_birth_place_county", ""),
            "writer_origin_farm": r.get("writer_origin_farm", ""),
            "writer_origin_municipality": r.get("writer_origin_municipality", ""),
            "writer_origin_county": r.get("writer_origin_county", ""),
            "text_plain": r["text_plain"],
            "text_html": r["text_html"],
            "images": images,
            "image_urls": r["image_urls"],
        }

def count_images(records):
    return sum(r["image_count"] for r in records)

DATASET_CARD = """\
---
license: cc-by-4.0
language:
- is
size_categories:
- 1K<n<10K
tags:
- icelandic
- handwriting
- ocr
- historical
- 19th-century
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*
  - split: validation
    path: validation/*
  - split: test
    path: test/*
---

# 19th Century Icelandic Letters OCR Benchmark

Handwritten letter dataset from [Bréfasafn Árnastofnunar](https://brefasafn.arnastofnun.is/).

## Dataset

- **Source**: [Bréfasafn 19. aldar](https://brefasafn.arnastofnun.is/) — Árni Magnússon Institute for Icelandic Studies
- **Size**: ~1,640 handwritten letters from ~350 writers
- **Images**: Full-resolution color scans (3264×2176 px JPG), 0–12 images per letter
- **Text**: Diplomatic transcriptions with TEI-like markup, plus cleaned plain text
- **License**: CC BY 4.0

## Features

| Column | Type | Description |
|--------|------|-------------|
| `id` | int32 | Letter ID on source site |
| `filename` | string | Archive filename |
| `date` | string | Letter date |
| `place_farm` | string | Writing location — farm |
| `place_municipality` | string | Writing location — municipality |
| `place_county` | string | Writing location — county |
| `note` | string | Editorial note |
| `collection` | string | Source archive |
| `shelfmark` | string | Shelfmark in archive |
| `recipient_name` | string | Recipient name |
| `recipient_title` | string | Recipient title/occupation |
| `image_status` | string | Note about image condition |
| `writer_name` | string | Writer name |
| `writer_title` | string | Writer title/occupation |
| `writer_gender` | string | Writer gender |
| `writer_birth_date` | string | Writer birth date |
| `writer_death_date` | string | Writer death date |
| `writer_birth_place_farm` | string | Writer birthplace — farm |
| `writer_birth_place_municipality` | string | Writer birthplace — municipality |
| `writer_birth_place_county` | string | Writer birthplace — county |
| `writer_origin_farm` | string | Writer origin — farm |
| `writer_origin_municipality` | string | Writer origin — municipality |
| `writer_origin_county` | string | Writer origin — county |
| `text_plain` | string | Clean plain text transcription |
| `text_html` | string | Raw HTML transcription with markup |
| `images` | List[Image] | List of PIL Images (full-res scans) |
| `image_urls` | List[str] | Source URLs for the images |

## Splits

80/10/10 train/validation/test, stratified by writer to prevent data leakage.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Sigurdur/19th-century-icelandic-letters", split="train")
print(dataset[0]["text_plain"])
dataset[0]["images"][0]
```

## Citation

```
@software{19th_century_icelandic_letters,
  author = {Sigurður Haukur},
  title = {19th Century Icelandic Letters OCR Benchmark},
  year = {2026},
  url = {https://huggingface.co/datasets/Sigurdur/19th-century-icelandic-letters}
}
```

Also cite the original source: Bréfasafn 19. aldar, Stofnun Árna Magnússonar í íslenskum fræðum. https://brefasafn.arnastofnun.is/
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-parquet", action="store_true")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--num-proc", type=int, default=1,
                        help="Parallel processes for image loading (default: 1)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Upload worker threads (default: 4)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.from_parquet:
        if not PARQUET_DIR.exists():
            print(f"Error: {PARQUET_DIR}/ does not exist")
            exit(1)
        print(f"Skipping dataset build, uploading existing {PARQUET_DIR}/")
    else:
        print("Loading records...")
        with open(RECORDS_PATH) as f:
            records = json.load(f)
        print(f"  {len(records)} records loaded")

        train_recs, val_recs, test_recs = split_records(records)
        if args.debug:
            train_recs = train_recs[:10]
            val_recs = val_recs[:10]
            test_recs = test_recs[:10]

        total_images = sum(count_images(s) for s in [train_recs, val_recs, test_recs])
        print(f"  Train: {len(train_recs):>5} records / {count_images(train_recs):>5} images")
        print(f"  Val:   {len(val_recs):>5} records / {count_images(val_recs):>5} images")
        print(f"  Test:  {len(test_recs):>5} records / {count_images(test_recs):>5} images")
        print(f"  Total: {sum(len(s) for s in [train_recs, val_recs, test_recs]):>5} records / {total_images:>5} images")

        features = make_features()

        if PARQUET_DIR.exists():
            shutil.rmtree(PARQUET_DIR)

        readme = PARQUET_DIR / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(DATASET_CARD)
        tqdm.write("  Created dataset card (README.md)")

        phases = [(name, recs) for name, recs in [
            ("train", train_recs), ("validation", val_recs), ("test", test_recs)
        ]]

        pbar = tqdm(phases, position=0, desc="Pipeline")
        for name, recs in phases:
            pbar.set_description(f"Building {name} ({len(recs)} records, {count_images(recs)} images)")

            split_dir = PARQUET_DIR / name
            split_dir.mkdir(parents=True, exist_ok=True)

            ds = Dataset.from_generator(
                record_generator,
                features=features,
                gen_kwargs={"records": recs},
                num_proc=args.num_proc,
            )

            ds.save_to_disk(str(split_dir), max_shard_size=args.max_shard_size)

            n_shards = len(list(split_dir.rglob("*.arrow")))
            total_mb = sum(f.stat().st_size for f in split_dir.rglob("*") if f.is_file()) / 1e6
            tqdm.write(f"  {name}: {n_shards} shards, {total_mb:.0f} MB")

            del ds
            pbar.update()

        tqdm.write("")
        total_gb = sum(f.stat().st_size for f in PARQUET_DIR.rglob("*") if f.is_file()) / 1e9
        tqdm.write(f"Total size: {total_gb:.2f} GB")

    tqdm.write("")
    tqdm.write(f"Uploading {PARQUET_DIR}/ to {REPO_ID} ...")
    tqdm.write("  (set HF_XET_HIGH_PERFORMANCE=1 before running for max speed)")

    api = HfApi()
    api.upload_large_folder(
        folder_path=str(PARQUET_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        num_workers=args.num_workers,
    )
    tqdm.write("\nDone!")
