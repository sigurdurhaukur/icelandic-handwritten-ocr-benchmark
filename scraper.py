import re
import time
import io
import json
from pathlib import Path
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from PIL import Image
from datasets import Dataset, DatasetDict, Image as HFImage, Features, Value, Sequence
from tqdm import tqdm

BASE_URL = "https://brefasafn.arnastofnun.is"
LISTING_URL = f"{BASE_URL}/leit_bref.php?order_by=skra_nafn&rod=ASC"

MIN_DELAY = 1.0
MAX_RETRIES = 3
OUTPUT_DIR = Path("data")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "IcelandicHandwrittenOCRBenchmark/0.1 (research)"
})

def fetch(url, delay=True, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            if delay:
                time.sleep(MIN_DELAY)
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries - 1:
                wait = MIN_DELAY * (attempt + 1) * 2
                print(f"  Retry {attempt + 1}/{retries} for {url} after {wait:.0f}s: {e}")
                time.sleep(wait)
            else:
                raise
    return None

def get_letter_ids():
    html = fetch(LISTING_URL, delay=False).text
    ids = sorted(set(int(x) for x in re.findall(r"leshamur\.php\?id=(\d+)", html)))
    return ids

def parse_letter_page(html):
    soup = BeautifulSoup(html, "lxml")
    meta = {}

    tables = soup.find_all("table", class_="CSSTableGenerator_leshamur")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            tds = row.find_all("td")
            if len(tds) >= 2:
                key = tds[0].get_text(strip=True).rstrip(":")
                val = tds[1].get_text(strip=True)
                meta[key] = val

    text_html = ""
    text_plain = ""
    for table in tables:
        first_row_text = table.find("tr").get_text(strip=True) if table.find("tr") else ""
        if "Texti bréfs" in first_row_text:
            content_rows = table.find_all("tr")[1:]
            for row in content_rows:
                tds = row.find_all("td")
                if tds:
                    content_td = tds[-1]
                    text_html = str(content_td)
                    text_plain = content_td.get_text(separator="\n", strip=True)
                    break

    image_urls = []
    for a_tag in soup.find_all("a", onclick=True):
        onclick = a_tag.get("onclick", "")
        m = re.search(r"popitup\('([^']+)'\)", onclick)
        if m:
            path = m.group(1)
            if "bref_myndir/" in path:
                full_url = BASE_URL + "/" + path.lstrip("/")
                image_urls.append(full_url)

    return meta, text_html, text_plain, image_urls

def english_key(icelandic_key):
    mapping = {
        "Nafn skrár": "filename",
        "Skráarnafn eldra": "old_filename",
        "Dagsetning": "date",
        "Ritunarstaður (bær)": "place_farm",
        "Ritunarstaður (Sveitarf.)": "place_municipality",
        "Ritunarstaður (Sýsla)": "place_county",
        "Athugasemd": "note",
        "Safn": "collection",
        "Safnmark": "shelfmark",
        "Nafn viðtakanda": "recipient_name",
        "Titill viðtakanda": "recipient_title",
        "Mynd": "image_status",
        "Bréfritari": "writer_name",
        "Titill bréfritara": "writer_title",
        "Kyn": "writer_gender",
        "Fæðingardagur": "writer_birth_date",
        "Dánardagur": "writer_death_date",
        "Fæðingarstaður (bær)": "writer_birth_place_farm",
        "Fæðingarstaður (sveitarf.)": "writer_birth_place_municipality",
        "Fæðingarstaður (sýsla)": "writer_birth_place_county",
        "Upprunaslóðir (bær)": "writer_origin_farm",
        "Upprunaslóðir (sveitarf.)": "writer_origin_municipality",
        "Upprunaslóðir (sýsla)": "writer_origin_county",
    }
    return mapping.get(icelandic_key, icelandic_key)

def download_image(url):
    try:
        resp = fetch(url, delay=True)
        img = Image.open(io.BytesIO(resp.content))
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return None

def save_checkpoint(records, failed_ids, completed_ids):
    checkpoint = {
        "completed_ids": list(completed_ids),
        "failed_ids": failed_ids,
        "records_meta": [{k: v for k, v in r.items() if k != "images"} for r in records],
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return None

def scrape_all(resume=False):
    print("Step 1: Getting letter IDs from listing page...")
    all_ids = get_letter_ids()
    print(f"  Found {len(all_ids)} letters (IDs {all_ids[0]}–{all_ids[-1]})")

    all_records = []
    failed_ids = []
    completed_ids = set()

    if resume:
        cp = load_checkpoint()
        if cp:
            all_records = cp.get("records_meta", [])
            for r in all_records:
                r["images"] = []
            completed_ids = set(cp.get("completed_ids", []))
            failed_ids = cp.get("failed_ids", [])
            print(f"  Resuming from checkpoint: {len(completed_ids)} completed, {len(failed_ids)} failed")

    remaining = [lid for lid in all_ids if lid not in completed_ids]
    print(f"\nStep 2: Scraping {len(remaining)} letter pages...")

    for idx, lid in enumerate(tqdm(remaining)):
        try:
            resp = fetch(f"{BASE_URL}/leshamur.php?id={lid}", delay=True)
            meta, text_html, text_plain, image_urls = parse_letter_page(resp.text)

            record = {"id": lid}
            for ik, v in meta.items():
                record[english_key(ik)] = v

            record["text_plain"] = text_plain
            record["text_html"] = text_html
            record["image_urls"] = image_urls
            record["image_count"] = len(image_urls)
            record["images"] = []

            all_records.append(record)
            completed_ids.add(lid)

            if len(completed_ids) % 100 == 0:
                save_checkpoint(all_records, failed_ids, completed_ids)

        except Exception as e:
            print(f"\n  Failed to scrape ID {lid}: {e}")
            failed_ids.append(lid)

    save_checkpoint(all_records, failed_ids, completed_ids)

    print(f"\n  Successfully scraped: {len(all_records)}")
    print(f"  Failed: {len(failed_ids)}")
    if failed_ids:
        print(f"  Failed IDs: {failed_ids}")

    print(f"\nStep 3: Downloading images for {len(all_records)} letters...")
    for record in tqdm(all_records):
        images = []
        for url in record["image_urls"]:
            img = download_image(url)
            if img is not None:
                images.append(img)
        record["images"] = images

    return all_records

def build_dataset(records):
    print("\nStep 4: Building HuggingFace dataset with 80/10/10 split by writer...")

    writers = defaultdict(list)
    for r in records:
        w = r.get("writer_name", "unknown")
        writers[w].append(r)

    writer_items = list(writers.items())
    writer_items.sort(key=lambda x: len(x[1]), reverse=True)

    train_recs = []
    val_recs = []
    test_recs = []

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

    print(f"  Train: {len(train_recs)}")
    print(f"  Val:   {len(val_recs)}")
    print(f"  Test:  {len(test_recs)}")

    def combine(recs):
        return {
            "id": [r["id"] for r in recs],
            "filename": [r.get("filename", "") for r in recs],
            "old_filename": [r.get("old_filename", "") for r in recs],
            "date": [r.get("date", "") for r in recs],
            "place_farm": [r.get("place_farm", "") for r in recs],
            "place_municipality": [r.get("place_municipality", "") for r in recs],
            "place_county": [r.get("place_county", "") for r in recs],
            "note": [r.get("note", "") for r in recs],
            "collection": [r.get("collection", "") for r in recs],
            "shelfmark": [r.get("shelfmark", "") for r in recs],
            "recipient_name": [r.get("recipient_name", "") for r in recs],
            "recipient_title": [r.get("recipient_title", "") for r in recs],
            "image_status": [r.get("image_status", "") for r in recs],
            "writer_name": [r.get("writer_name", "") for r in recs],
            "writer_title": [r.get("writer_title", "") for r in recs],
            "writer_gender": [r.get("writer_gender", "") for r in recs],
            "writer_birth_date": [r.get("writer_birth_date", "") for r in recs],
            "writer_death_date": [r.get("writer_death_date", "") for r in recs],
            "writer_birth_place_farm": [r.get("writer_birth_place_farm", "") for r in recs],
            "writer_birth_place_municipality": [r.get("writer_birth_place_municipality", "") for r in recs],
            "writer_birth_place_county": [r.get("writer_birth_place_county", "") for r in recs],
            "writer_origin_farm": [r.get("writer_origin_farm", "") for r in recs],
            "writer_origin_municipality": [r.get("writer_origin_municipality", "") for r in recs],
            "writer_origin_county": [r.get("writer_origin_county", "") for r in recs],
            "text_plain": [r["text_plain"] for r in recs],
            "text_html": [r["text_html"] for r in recs],
            "images": [r["images"] for r in recs],
            "image_urls": [r["image_urls"] for r in recs],
        }

    features = Features({
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

    dataset = DatasetDict({
        "train": Dataset.from_dict(combine(train_recs), features=features),
        "validation": Dataset.from_dict(combine(val_recs), features=features),
        "test": Dataset.from_dict(combine(test_recs), features=features),
    })

    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    args = parser.parse_args()

    records = scrape_all(resume=args.resume)

    meta_path = OUTPUT_DIR / "scraped_records.json"
    with open(meta_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "images"} for r in records], f, indent=2, ensure_ascii=False)
    print(f"\n  Saved metadata to {meta_path}")

    dataset = build_dataset(records)

    dataset_path = OUTPUT_DIR / "hf_dataset"
    dataset.save_to_disk(str(dataset_path))
    print(f"\n  Saved dataset to {dataset_path}")

    if args.push_to_hub:
        dataset.push_to_hub("Sigurdur/19th-century-icelandic-letters", private=False)
        print("\n  Pushed to HuggingFace Hub: Sigurdur/19th-century-icelandic-letters")
