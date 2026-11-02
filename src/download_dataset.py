"""Download the UCI SMS Spam Collection dataset programmatically."""

import os
import urllib.request
import zipfile
import shutil

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = os.path.join(RAW_DIR, "smsspamcollection.zip")
EXTRACTED_FILE = os.path.join(RAW_DIR, "SMSSpamCollection")


def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(EXTRACTED_FILE):
        print(f"Dataset already present at {EXTRACTED_FILE}")
        return EXTRACTED_FILE

    print(f"Downloading from {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print("Download complete. Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
    # Clean up zip
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if not os.path.exists(EXTRACTED_FILE):
        # find the extracted file
        for root, _, files in os.walk(RAW_DIR):
            for f in files:
                if "SMSSpamCollection" in f:
                    shutil.move(os.path.join(root, f), EXTRACTED_FILE)
    print(f"Dataset saved to {EXTRACTED_FILE}")
    return EXTRACTED_FILE


if __name__ == "__main__":
    download_dataset()