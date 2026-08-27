import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://feedsdatabase.ilri.org"
SEARCH_PAGES = [
    "/concentrate-feeds",
    "/herbaceous-forages",
    "/fodder-trees-and-shrubs",
    "/food-crops-cereals-legumes-green",
    "/food-crops-cereals-legumes-residues",
    "/food-crops-roots-tubers",
    "/food-crops-others",
    "/mineral-supplements",
    "/other-less-common-feeds",
]

OUT = Path("datasets/extracted/feed_nutrition")
OUT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

nutrition_fields = {
    "ADF": "ADF",
    "ADL": "ADL",
    "CP": "CP",
    "DM": "DM",
    "IVDMD": "IVDMD",
    "ME": "ME",
    "NDF": "NDF",
    "OM": "OM",
    "NEm": "NEm",
    "NEg": "NEg",
    "NEl": "NEl",
    "Ca": "Ca",
    "P": "P",
    "Fe": "Fe",
    "K": "K",
    "Mg": "Mg",
    "Mn": "Mn",
    "Na": "Na",
    "Zn": "Zn",
}

all_urls = set()

print("Collecting feed pages...")

for page in SEARCH_PAGES:
    url = urljoin(BASE, page)

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print("FAILED CATEGORY:", page, exc)
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Feed detail pages are /node/<number>
        if re.fullmatch(r"/node/\d+", href):
            all_urls.add(urljoin(BASE, href))

print("Feed pages found:", len(all_urls))

records = []

for i, url in enumerate(sorted(all_urls), 1):

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"[{i}/{len(all_urls)}] FAILED:", url, exc)
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    text = soup.get_text("\n", strip=True)
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    row = {
        "source_url": url
    }

    # Feed title
    title = soup.find("h1")
    row["feed_name"] = title.get_text(" ", strip=True) if title else None

    # Extract "Field: Value" style lines
    for field, output_name in nutrition_fields.items():

        value = None

        for idx, line in enumerate(lines):
            if line.lower().startswith(field.lower() + " "):
                parts = re.split(r":\s*", line, maxsplit=1)

                if len(parts) == 2:
                    value = parts[1].strip()

            elif line.lower() == field.lower() and idx + 1 < len(lines):
                value = lines[idx + 1].strip()

        row[output_name] = value

    # Country
    for idx, line in enumerate(lines):
        if line.lower() == "country:" and idx + 1 < len(lines):
            row["country"] = lines[idx + 1]
        elif line.lower().startswith("country:"):
            row["country"] = line.split(":", 1)[1].strip()

    # Location
    for idx, line in enumerate(lines):
        if line.lower() == "location:" and idx + 1 < len(lines):
            row["location"] = lines[idx + 1]

    # Feed type
    for idx, line in enumerate(lines):
        if line.lower() == "feed type:" and idx + 1 < len(lines):
            row["feed_type"] = lines[idx + 1]

    # Keep only pages containing at least one nutrition value
    if any(row.get(k) not in (None, "") for k in nutrition_fields.values()):
        records.append(row)

    if i % 100 == 0:
        print(f"Processed {i}/{len(all_urls)}")

    time.sleep(0.05)

df = pd.DataFrame(records)

print("\nCollected records:", len(df))

if df.empty:
    raise RuntimeError(
        "No nutrition records were extracted. "
        "Inspect one feed page manually before continuing."
    )

# Convert nutritional columns to numeric
for col in nutrition_fields.values():
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+)")[0]
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Moisture from DM
if "DM" in df.columns:
    df["Moisture"] = 100 - df["DM"]

# Remove exact duplicates
df = df.drop_duplicates()

output = OUT / "feed_composition.csv"
df.to_csv(output, index=False)

print("\nSaved:")
print(output)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nNon-null counts:")
print(df[nutrition_fields.values()].notna().sum().sort_values(ascending=False))