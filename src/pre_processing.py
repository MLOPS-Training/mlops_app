from lemmatizer import Lemmatizer

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
import re


# paths
PRE_CURATED_DATA_DIR = Path("./src/data/pre_curated")
CURATED_DATA_DIR = Path("./src/data/curated")
OUTPUT_CHART_PATH = Path("./src/static/personality_pie_chart.html")
OUTPUT_WORD_CLOUD_IMAGE_PATH = Path("./src/static/word_cloud.png")


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


def pre_processed_csv(file_path: Path):
    file_name = file_path.stem  # Use the name without extension

    try:
        mbti = pd.read_csv(file_path, encoding="ISO-8859-1")
    except Exception:
        print("End of all files")
        return

    # Clean the text
    mbti["posts"] = clear_text(mbti)

    # Plot personality type pie chart using Plotly Express
    fig = px.pie(mbti, names="type", title="Personality type", hole=0.3)
    fig.write_html(OUTPUT_CHART_PATH)

    # Create a Matplotlib subplot
    fig_subplots = make_subplots(rows=1, cols=1)

    # Iterate over each trace in the Plotly figure and add it to the subplot
    for trace in fig["data"]:
        if trace.type == "pie":
            # Create a subplot for the pie chart
            fig_subplots.add_trace(trace)

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
        plt.savefig(OUTPUT_WORD_CLOUD_IMAGE_PATH, bbox_inches="tight")
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
    curated_parquet_path = CURATED_DATA_DIR.joinpath("curated_data.parquet")

    if not curated_parquet_path.exists():
        # If the parquet file doesn't exist, create it
        table = pa.Table.from_pandas(mbti)
        with pq.ParquetWriter(str(curated_parquet_path), table.schema) as writer:
            writer.write_table(table)
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
