import subprocess
from pathlib import Path
from tqdm import tqdm
import sys

# ----------------------------
# CONFIG
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URL_FILE = PROJECT_ROOT / "data" / "raw" / "video_urls.txt"
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
COOKIES_FILE = PROJECT_ROOT / "cookies.txt"   # MUST EXIST

# ----------------------------
# SAFETY CHECKS
# ----------------------------

if not URL_FILE.exists():
    print("Missing file:", URL_FILE)
    sys.exit(1)

if not COOKIES_FILE.exists():
    print("Missing cookies.txt at:", COOKIES_FILE)
    print("Export cookies from your browser and place it in the project root.")
    sys.exit(1)

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

with open(URL_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

urls = list(reversed(urls))[:140]


print(f"Found {len(urls)} video URLs.")
print("Using cookies from:", COOKIES_FILE)

# ----------------------------
# DOWNLOAD LOOP (WHISPER SAFE)
# ----------------------------

for i, url in enumerate(tqdm(urls, desc="Downloading"), start=1):
    output_path = VIDEO_DIR / f"{i:04d}.mp4"

    # Skip if already downloaded
    if output_path.exists():
        continue

    command = [
        "yt-dlp",
        "--cookies", str(COOKIES_FILE),
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
        "--no-playlist",
        "--no-warnings",
        "-o", str(output_path),
        url
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to download: {url}")

print("✅ All downloads complete.")
