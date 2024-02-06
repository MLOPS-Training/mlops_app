import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mlops_app.data_pipeline.pre_processing.pre_processing import pre_processed_csv
from mlops_app.data_pipeline.training_testing.train_test_LR import train_and_test_models as train_and_test_LR
from mlops_app.data_pipeline.training_testing.train_test_LSVC import train_and_test_models as train_and_test_LSVC

import hashlib

# Paths
RAW_DATA_DIR = Path('../data_pipeline/data/raw')
PRE_CURATED_DATA_DIR = Path('../data_pipeline/data/pre_curated')
CURATED_DATA_DIR = Path('../data_pipeline/data/curated')
os.makedirs(PRE_CURATED_DATA_DIR, exist_ok=True)
os.makedirs(CURATED_DATA_DIR, exist_ok=True)

# File to store processed file hashes
PROCESSED_FILE_HASHES_FILE = Path('./processed_file_hashes.txt')


def load_processed_file_hashes():
    """Load processed file hashes from the file."""
    if PROCESSED_FILE_HASHES_FILE.exists():
        with open(PROCESSED_FILE_HASHES_FILE, 'r') as file:
            return set(file.read().splitlines())
    return set()


def save_processed_file_hash(file_hash):
    """Save processed file hash to the file."""
    with open(PROCESSED_FILE_HASHES_FILE, 'a') as file:
        file.write(file_hash + '\n')


class FileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_message_time = time.time()

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"New file detected in the path data/raw: {event.src_path}")
        time.sleep(2)  # Wait for the file to be written
        pre_processed_csv(Path(event.src_path))

    def process_existing_files(self):
        """Process existing files when the script starts."""
        for file_path in RAW_DATA_DIR.glob("*.csv"):
            if file_path.is_file():
                print(f"Processing existing file: {file_path}")
                pre_processed_csv(file_path)

    def trigger_preprocessing(self):
        """Trigger preprocessing for curated_data.parquet."""
        pre_processed_csv(CURATED_DATA_DIR / 'curated_data.parquet')

    def on_modified(self, event):
        if event.is_directory:
            return
        print(f"File modified in the path data/raw: {event.src_path}")
        time.sleep(2)  # Wait for the file to be written
        pre_processed_csv(Path(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"File deleted in the path data/raw: {event.src_path}")
        # You can add additional logic for handling file deletion if needed


def monitor_directory():
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(RAW_DATA_DIR), recursive=False)
    observer.start()

    try:
        # Initialize processed file hashes set
        processed_files = load_processed_file_hashes()

        # Process existing files when the script starts
        event_handler.process_existing_files()

        while True:
            current_files = set(RAW_DATA_DIR.glob("*.csv"))
            new_files = current_files - processed_files

            for file_path in new_files:
                if file_path.is_file():
                    print(f"Processing new file: {file_path}")
                    pre_processed_csv(file_path)
                    file_hash = hashlib.md5(file_path.stem.encode('utf-8')).hexdigest()
                    save_processed_file_hash(file_hash)
                    processed_files.add(file_hash)

            current_time = time.time()

            # Trigger preprocessing and testing every 30 seconds
            if current_time - event_handler.last_message_time >= 30:
                event_handler.trigger_preprocessing()
                event_handler.last_message_time = current_time

                # After processing all new files, trigger the training_testing and testing
                print("Training and testing Logistic Regression model will begin soon...")
                train_and_test_LR('curated_data.parquet')
                print("Training and testing LSVC model will begin soon...")
                train_and_test_LSVC('curated_data.parquet')

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping the monitoring script...")
    observer.join()


if __name__ == "__main__":
    monitor_directory()
