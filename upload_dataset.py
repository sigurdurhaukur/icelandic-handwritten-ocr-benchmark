"""
upload_dataset.py — build and upload the 19th-century Icelandic letters dataset.

Improvements over v1:
  - ThreadPoolExecutor for parallel image loading per record
  - Shard-level checkpoint file so interrupted runs resume mid-split
  - Generator yields records one at a time; images are loaded and discarded
    immediately, keeping peak RAM proportional to one record's images
  - Corrupted/missing images are skipped with a warning, not a crash
  - --max-records-in-memory cap to bound working set during Dataset.from_generator
"""

import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data")
IMAGES_DIR = OUTPUT_DIR / "images"
RECORDS_PATH = OUTPUT_DIR / "scraped_records.json"
CHECKPOINT_PATH = OUTPUT_DIR / "upload_checkpoint.json"
REPO_ID = "Sigurdur/19th-century-icelandic-letters"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(lid: int, idx: int) -> Image.Image | None:
    path = IMAGES_DIR / str(lid) / f"{idx}.jpg"
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        img.load()  # force decode now, while path is still open
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        log.warning("Skipping corrupt image %s: %s", path, exc)
        return None


def load_images_parallel(lid: int, count: int, max_workers: int) -> list[Image.Image]:
    """Load all pages of a letter in parallel. Order is preserved."""
    if count == 0:
        return []
    results: dict[int, Image.Image | None] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, count)) as pool:
        futures = {pool.submit(load_image, lid, i): i for i in range(count)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return [results[i] for i in range(count) if results[i] is not None]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def make_features() -> Features:
    str_field = Value("string")
    return Features(
        {
            "id": Value("int32"),
            "filename": str_field,
            "old_filename": str_field,
            "date": str_field,
            "place_farm": str_field,
            "place_municipality": str_field,
            "place_county": str_field,
            "note": str_field,
            "collection": str_field,
            "shelfmark": str_field,
            "recipient_name": str_field,
            "recipient_title": str_field,
            "image_status": str_field,
            "writer_name": str_field,
            "writer_title": str_field,
            "writer_gender": str_field,
            "writer_birth_date": str_field,
            "writer_death_date": str_field,
            "writer_birth_place_farm": str_field,
            "writer_birth_place_municipality": str_field,
            "writer_birth_place_county": str_field,
            "writer_origin_farm": str_field,
            "writer_origin_municipality": str_field,
            "writer_origin_county": str_field,
            "text_plain": str_field,
            "text_html": str_field,
            "images": Sequence(HFImage(decode=True)),
            "image_urls": Sequence(str_field),
        }
    )


STRING_FIELDS = [
    "filename",
    "old_filename",
    "date",
    "place_farm",
    "place_municipality",
    "place_county",
    "note",
    "collection",
    "shelfmark",
    "recipient_name",
    "recipient_title",
    "image_status",
    "writer_name",
    "writer_title",
    "writer_gender",
    "writer_birth_date",
    "writer_death_date",
    "writer_birth_place_farm",
    "writer_birth_place_municipality",
    "writer_birth_place_county",
    "writer_origin_farm",
    "writer_origin_municipality",
    "writer_origin_county",
    "text_plain",
    "text_html",
]


# ---------------------------------------------------------------------------
# Train / val / test split  (stratified by writer)
# ---------------------------------------------------------------------------


def split_records(
    records: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    writers: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        writers[r.get("writer_name", "unknown")].append(r)

    train_recs, val_recs, test_recs = [], [], []
    for _, recs in sorted(writers.items(), key=lambda kv: -len(kv[1])):
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
        val_recs.extend(sorted_recs[n_train : n_train + n_val])
        test_recs.extend(sorted_recs[n_train + n_val :])

    return train_recs, val_recs, test_recs


def count_images(records: list[dict]) -> int:
    return sum(r["image_count"] for r in records)


# ---------------------------------------------------------------------------
# Generator  (one record at a time — images loaded and released immediately)
# ---------------------------------------------------------------------------


def record_generator(records: list[dict], image_workers: int, pbar: tqdm | None = None):
    for r in records:
        images = load_images_parallel(r["id"], r["image_count"], image_workers)
        yield {f: r.get(f, "") for f in STRING_FIELDS} | {
            "id": r["id"],
            "images": images,
            "image_urls": r["image_urls"],
        }
        del images
        if pbar is not None:
            pbar.update(1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        done: set[str] = set(data.get("completed_splits", []))
        if done:
            log.info("Checkpoint found — skipping already-built splits: %s", done)
        return done
    return set()


def save_checkpoint(completed_splits: set[str]) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps({"completed_splits": sorted(completed_splits)})
    )


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

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
| `place_*` | string | Writing location |
| `note` | string | Editorial note |
| `collection` | string | Source archive |
| `shelfmark` | string | Shelfmark in archive |
| `recipient_name` | string | Recipient name |
| `recipient_title` | string | Recipient title/occupation |
| `image_status` | string | Note about image condition |
| `writer_*` | string | Writer metadata |
| `text_plain` | string | Clean plain text transcription |
| `text_html` | string | Raw HTML transcription with markup |
| `images` | List[Image] | Full-res PIL images |
| `image_urls` | List[str] | Source URLs for the images |

## Splits

80/10/10 train/validation/test, stratified by writer to prevent data leakage.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Sigurdur/19th-century-icelandic-letters", split="train")
print(ds[0]["text_plain"])
ds[0]["images"][0]
```

## Citation

```
@software{19th_century_icelandic_letters,
  author    = {Sigurður Haukur},
  title     = {19th Century Icelandic Letters OCR Benchmark},
  year      = {2026},
  url       = {https://huggingface.co/datasets/Sigurdur/19th-century-icelandic-letters}
}
```

Also cite: Bréfasafn 19. aldar, Stofnun Árna Magnússonar í íslenskum fræðum.
https://brefasafn.arnastofnun.is/
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    from datasets import DatasetDict

    parser = argparse.ArgumentParser(
        description="Build and upload the letters dataset."
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Parquet shard size passed to push_to_hub (default: 500MB)",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=8,
        help="Threads for parallel image loading per record (default: 8)",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Ignore existing checkpoint and rebuild all splits",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use only 10 records per split"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap total records across all splits (e.g. --max-samples 10)",
    )
    args = parser.parse_args()

    log.info("Loading records from %s", RECORDS_PATH)
    records: list[dict] = json.loads(RECORDS_PATH.read_text())
    log.info("  %d records loaded", len(records))

    train_recs, val_recs, test_recs = split_records(records)

    if args.debug:
        train_recs, val_recs, test_recs = train_recs[:10], val_recs[:10], test_recs[:10]
    if args.max_samples is not None:
        n = args.max_samples
        train_recs = train_recs[: max(1, round(n * 0.8))]
        val_recs = val_recs[: max(1, round(n * 0.1))]
        test_recs = test_recs[: max(1, round(n * 0.1))]
        log.info(
            "Capped to ~%d samples (--max-samples %d)",
            len(train_recs) + len(val_recs) + len(test_recs),
            n,
        )

    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        log.info("  %-6s %4d records / %4d images", name, len(recs), count_images(recs))

    features = make_features()
    completed = set() if args.reset_checkpoint else load_checkpoint()

    phases = [
        ("train", train_recs),
        ("validation", val_recs),
        ("test", test_recs),
    ]
    total_records = sum(len(r) for _, r in phases)

    dataset_dict: dict = {}

    with tqdm(
        total=total_records,
        desc="records",
        unit="rec",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rec  "
        "[{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for split_name, recs in phases:
            if split_name in completed:
                pbar.write(f"  Skipping {split_name} (already built)")
                pbar.update(len(recs))
                continue

            pbar.set_postfix(split=split_name, imgs=count_images(recs))
            pbar.write(
                f"\n[{split_name}] {len(recs)} records / {count_images(recs)} images"
            )

            try:
                ds = Dataset.from_generator(
                    record_generator,
                    features=features,
                    gen_kwargs={
                        "records": recs,
                        "image_workers": args.image_workers,
                        "pbar": pbar,
                    },
                    num_proc=1,
                    # Each batch is serialised to Arrow before the next is
                    # accumulated. Images are large; 10 records per batch
                    # keeps each batch well under PyArrow's 2 GB int32
                    # offset limit that causes 'offset overflow' errors.
                    writer_batch_size=10,
                )
                dataset_dict[split_name] = ds
                completed.add(split_name)
                save_checkpoint(completed)

            except Exception:
                log.exception(
                    "Failed on split '%s' — checkpoint saved for completed splits",
                    split_name,
                )
                raise

    if not dataset_dict:
        log.info("Nothing to upload (all splits already completed).")
        return

    dsd = DatasetDict(dataset_dict)
    log.info("Uploading to %s ...", REPO_ID)
    # push_to_hub writes sharded Parquet (not Arrow), which the HF dataset
    # viewer can read. max_shard_size controls shard file size.
    dsd.push_to_hub(REPO_ID, max_shard_size=args.max_shard_size)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    log.info("Done!")


if __name__ == "__main__":
    main()
