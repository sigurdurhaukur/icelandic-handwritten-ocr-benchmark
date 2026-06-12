# 19th Century Icelandic Letters OCR Benchmark

Handwritten letter dataset from [Bréfasafn Árnastofnunar](https://brefasafn.arnastofnun.is/).

## Dataset

- **Source**: [Bréfasafn 19. aldar](https://brefasafn.arnastofnun.is/) — Árni Magnússon Institute for Icelandic Studies
- **Size**: ~1,640 handwritten letters from ~350 writers
- **Images**: Full-resolution color scans (3264×2176 px JPG), 0–12 images per letter
- **Text**: Diplomatic transcriptions with TEI-like markup, plus cleaned plain text
- **License**: Images and metadata are CC BY 4.0 (per Árnastofnun policy). This compiled dataset is released for research purposes. Please cite the original source.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int32 | Letter ID on source site |
| `filename` | string | Archive filename (e.g., `AdaBja-1874-00-00`) |
| `date` | string | Letter date (e.g., `A-1874-00-00`) |
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

80/10/10 train/validation/test, **stratified by writer** to prevent data leakage (same writer never appears in multiple splits).

## Citation

If you use this dataset, please cite:

```
@software{19th_century_icelandic_letters,
  author = {Sigurður Haukur},
  title = {19th Century Icelandic Letters OCR Benchmark},
  year = {2026},
  url = {https://huggingface.co/datasets/Sigurdur/19th-century-icelandic-letters}
}
```

Also cite the original source:

```
Bréfasafn 19. aldar. Stofnun Árna Magnússonar í íslenskum fræðum.
https://brefasafn.arnastofnun.is/
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Sigurdur/19th-century-icelandic-letters", split="train")
print(dataset[0]["text_plain"])
dataset[0]["images"][0]  # PIL Image
```

## Scraping

Run `python scraper.py` to reproduce (takes ~3 hours due to 1 req/s rate limit).
