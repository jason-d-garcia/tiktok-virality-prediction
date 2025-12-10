import pandas as pd
import subprocess
import json
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "tiktok_stats.csv"
URL_FILE = PROJECT_ROOT / "data" / "raw" / "video_urls.txt"

N = 138  # number of videos to keep

# ----------------------------
# LOAD CSV (SKIP SCHEMA ROW)
# ----------------------------

df = pd.read_csv(CSV_PATH)

# ✅ Drop schema row if it's accidentally duplicated as row 0
if isinstance(df.iloc[0, 0], str):
    print("⚠️ Detected schema row inside data → dropping first row")
    df = df.iloc[1:].reset_index(drop=True)

# ✅ Keep ONLY first 138 rows
df = df.iloc[:N].reset_index(drop=True)

# ----------------------------
# LOAD + REVERSE URLS
# ----------------------------

with open(URL_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

# ✅ Reverse and take FIRST 138
urls = list(reversed(urls))[:N]

assert len(df) == len(urls), "❌ CSV rows and URL count do NOT match after trimming!"

# ----------------------------
# ADD DESCRIPTION COLUMN IF MISSING
# ----------------------------

if "description" not in df.columns:
    df.insert(1, "description", "")
    print("✅ Added description column")

# ----------------------------
# METADATA SCRAPE LOOP
# ----------------------------

descriptions = []

for url in tqdm(urls, desc="Fetching descriptions"):
    try:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--dump-json",
            url
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        metadata = json.loads(result.stdout)
        desc = metadata.get("description", "")

    except Exception:
        print(f"Failed to fetch: {url}")
        desc = ""

    descriptions.append(desc)

# ----------------------------
# ASSIGN + SAVE
# ----------------------------

df["description"] = descriptions
df.to_csv(CSV_PATH, index=False)

print("Descriptions added for first 138 videos.")
print("CSV saved to:", CSV_PATH)