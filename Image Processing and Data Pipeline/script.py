import os
import requests
from PIL import Image
from io import BytesIO
import csv
from datetime import datetime
import subprocess
import shutil

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_FILE = os.path.join(BASE_DIR, "metadata.csv")
REPORT_FILE = os.path.join(BASE_DIR, "report.txt")
URLS_FILE = os.path.join(BASE_DIR, "urls")

RESIZE_DIMENSION = (800, 600)
IMAGE_QUALITY = 95
URLS_FILE = os.path.join(os.path.dirname(__file__), "urls")
# -----------------------------
# READ URLS (handles Python-style urls file)
# -----------------------------
IMAGE_URLS = []


with open(URLS_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if (
            not line
            or line.startswith("IMAGE_URLS")
            or line == "["
            or line == "]"
        ):
            continue

        line = line.strip('",')
        IMAGE_URLS.append(line)

# -----------------------------
# SETUP
# -----------------------------
os.makedirs(IMAGE_DIR, exist_ok=True)

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Filename",
        "Original Size",
        "Final Size",
        "Download Time",
        "Image URL"
    ])

CATIMG_AVAILABLE = shutil.which("catimg") is not None
success_count = 0

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for index, url in enumerate(IMAGE_URLS, start=1):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        original_size = image.size

        image = image.convert("RGB")
        image = image.resize(RESIZE_DIMENSION)

        filename = f"image_{index}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)

        image.save(filepath, quality=IMAGE_QUALITY)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,
                original_size,
                image.size,
                timestamp,
                url
            ])

        if CATIMG_AVAILABLE:
            subprocess.run(["catimg", filepath])

        success_count += 1

    except Exception as e:
        print(f"Failed to process image {index}: {e}")

# -----------------------------
# REPORT
# -----------------------------
with open(REPORT_FILE, "w") as f:
    f.write("IMAGE PROCESSING & DATA PIPELINE REPORT\n")
    f.write("======================================\n")
    f.write(f"Total URLs provided : {len(IMAGE_URLS)}\n")
    f.write(f"Images processed    : {success_count}\n")
    f.write(f"Resize dimension    : {RESIZE_DIMENSION[0]} x {RESIZE_DIMENSION[1]}\n")
    f.write(f"Image quality       : {IMAGE_QUALITY}%\n")
    f.write("Status              : Completed\n")

print("Pipeline finished")
