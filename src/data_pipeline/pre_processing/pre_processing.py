from utils.lemmatizer import Lemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import plotly.express as px
import pyarrow as pa
import pandas as pd
import hashlib
import os
import re


# Pathsapp
RAW_DATA_DIR = Path("./src/data_pipeline/data/raw")
PRE_CURATED_DATA_DIR = Path("./src/data_pipeline/data/pre_curated")
CURATED_DATA_DIR = Path("./src/data_pipeline/data/curated")
os.makedirs(PRE_CURATED_DATA_DIR, exist_ok=True)
os.makedirs(CURATED_DATA_DIR, exist_ok=True)

PROCESSED_FILE_HASHES_FILE = Path("./src/data_pipeline/processed_file_hashes.txt")


def clear_text(data):
    cleaned_text = []

    for sentence in tqdm(data.posts):
        # Convert to lowercase
        sentence = sentence.lower()

        # Remove URLs with various quotes
        sentence = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),\'\"]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
            sentence,
        )

        # Remove special characters except for certain punctuation and smileys
        sentence = re.sub(
            r'[^0-9a-zA-Z\s.,!?;:)(\[\]{}<>:-;\'"/\\^_^=+-]', " ", sentence
        )

        # Remove extra spaces
        sentence = re.sub(r"\s+", " ", sentence).strip()

        cleaned_text.append(sentence)

    return cleaned_text


def save_figure(figure, filename, output_folder="./src/static"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figure.number)  # Use figure directly without subscripting
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def load_processed_file_hashes():
    """Load processed file hashes from the file."""
    if PROCESSED_FILE_HASHES_FILE.exists():
        with open(PROCESSED_FILE_HASHES_FILE, "r") as file:
            return set(file.read().splitlines())
    return set()


# Add a set to keep track of processed file names
processed_files = load_processed_file_hashes()


def save_processed_file_hash(file_hash):
    """Save processed file hash to the file."""
    with open(PROCESSED_FILE_HASHES_FILE, "a") as file:
        file.write(file_hash + "\n")


def pre_processed_csv(file_path: Path):
    # Output results
    file_name = file_path.stem  # Use the name without extension

    # Check if the file has already been processed
    file_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()
    if file_hash in processed_files:
        print(f"File {file_name} has already been processed.")
        return

    try:
        mbti = pd.read_csv(file_path, encoding="ISO-8859-1")
    except Exception:
        print("End of all files")
        return

    # Clean the text
    mbti["posts"] = clear_text(mbti)

    # Plot personality type pie chart using Plotly Express
    fig = px.pie(mbti, names="type", title="Personality type", hole=0.3)

    # Save the Plotly figure as HTML (optional)
    output_folder = "./src/static"
    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    fig.write_html("./src/static/personality_pie_chart.html")

    # Create a Matplotlib subplot
    fig_subplots = make_subplots(rows=1, cols=1)

    # Iterate over each trace in the Plotly figure and add it to the subplot
    for trace in fig["data"]:
        if trace.type == "pie":
            # Create a subplot for the pie chart
            fig_subplots.add_trace(trace)

    # Display the Plotly figure if needed
    # fig_subplots.show()

    # TfidfVectorizer with Lemmatizer
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", tokenizer=Lemmatizer()
    )
    X = vectorizer.fit_transform(mbti["posts"])

    # Convert the TF-IDF matrix back to text
    text_data = [
        " ".join(vectorizer.inverse_transform(X[i])[0]) for i in range(X.shape[0])
    ]

    # Word Cloud
    if len(text_data) > 0:
        wc = WordCloud(max_words=400)
        wc.generate(" ".join(text_data))
        plt.figure(figsize=(20, 15))
        plt.axis("off")
        plt.imshow(wc)

        # Save the word cloud image directly
        plt.savefig("./src/static/word_cloud.png", bbox_inches="tight")
        plt.close()  # Close the figure to free up resources
    else:
        print("Not enough data to generate a word cloud.")

    now = pd.to_datetime("now").strftime("%Y%m%d%H%M%S")

    # Save the preprocessed data to the specified file_path in pre_curated folder
    processed_file_path = (
        PRE_CURATED_DATA_DIR / f"{now}-{file_name}-pre_curated_data.csv"
    )
    pd.DataFrame(mbti).to_csv(processed_file_path, index=False)
    print(f"Processed {file_name} : OK")

    # Append the preprocessed data to the curated parquet file
    curated_parquet_path = CURATED_DATA_DIR / "curated_data.parquet"

    if not curated_parquet_path.exists():
        # If the parquet file doesn't exist, create it
        table = pa.Table.from_pandas(mbti)
        with pq.ParquetWriter(str(curated_parquet_path), table.schema) as writer:
            writer.write_table(table)
        # show the 1025 element of the parquet file (just for testing that the parquet file is created correctly)
        print(pq.read_table(str(curated_parquet_path)).to_pandas().iloc[1025])
        print(f"Added {file_name} to new parquet file named : curated_data.parquet")
    else:
        # If the parquet file exists, read existing data and append the new data
        existing_data = pq.read_table(str(curated_parquet_path)).to_pandas()
        new_data = pd.concat([existing_data, mbti], ignore_index=True)

        # Write the combined data back to the Parquet file
        table = pa.Table.from_pandas(new_data)
        with pq.ParquetWriter(str(curated_parquet_path), table.schema) as writer:
            writer.write_table(table)
        print(f"Appended {file_name} to curated_data.parquet")

    # Add the processed file name to the set and save to file
    processed_files.add(file_hash)
    save_processed_file_hash(file_hash)


if __name__ == "__main__":
    # Get a list of all CSV files in the raw data folder
    mbti_files = [file for file in RAW_DATA_DIR.glob("*.csv")]

    for mbti_file_path in mbti_files:
        pre_processed_csv(mbti_file_path)