from utils.lemmatizer import Lemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

import pandas as pd
import joblib
import os


def save_model_and_results(model, X_test, y_test, model_name):
    # Save the model as a pickle file
    model_output_path_pickle = (
        Path(__file__).resolve().parent
        / "models"
        / "models_train"
        / f"{model_name}_model"
        / "train.pkl"
    )
    model_output_path_pickle.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, model_output_path_pickle)
        print(
            f"{model_name} Model saved successfully as pickle at: {
              model_output_path_pickle}"
        )
    except Exception as e:
        print(f"Error saving {model_name} model as pickle: {e}")

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Combine features, true labels, and predicted labels
    results_df = pd.DataFrame(
        {"True_Labels": y_test, "Predicted_Labels": test_predictions}
    )

    # Save the DataFrame to Parquet
    results_parquet_path = (
        Path(__file__).resolve().parent
        / "models"
        / "models_test"
        / f"{model_name}_model"
        / f"test_{model_name}.parquet"
    )
    results_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(results_parquet_path, index=False)

    print(
        f"Test predictions for {
          model_name} saved successfully at: {results_parquet_path}"
    )


def train_and_test_logistic_regression(train_data, test_data):
    # Vectorize text data
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", tokenizer=Lemmatizer()
    )
    X_train = vectorizer.fit_transform(train_data.posts)
    X_test = vectorizer.transform(test_data.posts)

    # Transform target labels to numerical values
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train_data.type)
    y_test = target_encoder.transform(test_data.type)

    # Train Logistic Regression model
    model_log = LogisticRegression(max_iter=3000, C=0.5, n_jobs=-1)
    model_log.fit(X_train, y_train)

    # Save Logistic Regression model
    save_model_and_results(model_log, X_test, y_test, "logistic_regression")


def train_and_test_models(data_filename):
    # Load the curated data
    curated_data_path = (
        Path(__file__).resolve().parent.parent / "data" / "curated" / data_filename
    )
    try:
        curated_data_modified_time = os.path.getmtime(curated_data_path)
        data = pd.read_parquet(curated_data_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Check if the curated_data.parquet file has been modified since the last training
    last_train_time_file = (
        Path(__file__).resolve().parent / "models" / "last_train_time.txt"
    )
    if last_train_time_file.exists():
        last_train_time = float(last_train_time_file.read_text())
        if curated_data_modified_time <= last_train_time:
            print(
                "No need to re-run training. The curated_data.parquet file hasn't been modified."
            )
            return

    # Stratify split to ensure equal distribution of data
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data.type
    )

    # Train and test Logistic Regression
    train_and_test_logistic_regression(train_data, test_data)


if __name__ == "__main__":
    # Replace 'curated_data.parquet' with your actual curated data file name
    train_and_test_models("curated_data.parquet")