from pre_processing import pre_processed_csv
from models import retrain_models

from pathlib import Path

import hashlib
import time
import os


# paths
RAW_DATA_DIR = Path("./src/data/raw")

# file to store processed file hashes
PROCESSED_FILE_HASHES_FILE = Path("./src/temp/processed_file_hashes.txt")


def check_if_file_processed(file_name):
    if PROCESSED_FILE_HASHES_FILE.exists():
        file_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()
        with open(PROCESSED_FILE_HASHES_FILE, "r") as file:
            return file_hash in set(file.read().splitlines())
    return False


def save_processed_file_hash(file_name):
    file_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()
    with open(PROCESSED_FILE_HASHES_FILE, "a") as file:
        file.write(file_hash + "\n")


def monitor_directory():
    while True:
        print("Monitoring for new files")
        new_file = False
        for file in RAW_DATA_DIR.glob("*.csv"):
            if not check_if_file_processed(file.name):
                os.environ["MODELS_UP_TO_DATE"] = "False"
                print(f"Processing new file: {file.name}")
                pre_processed_csv(RAW_DATA_DIR.joinpath(file.name))
                save_processed_file_hash(file.name)
                new_file = True

        # After processing all new files retrain models
        if new_file:
            retrain_models()
            os.environ["MODELS_UP_TO_DATE"] = "True"
        else:
            print("No new files found")

        # sleep for 1 minute
        time.sleep(60)
