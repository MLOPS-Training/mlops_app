from lemmatizer import Lemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from datetime import datetime
from threading import Thread
from pathlib import Path

import pandas as pd
import joblib


# paths
CURATED_DATA_PATH = Path("./src/data/curated/curated_data.parquet")
MODEL_RESULT_OUTPUT_PATH = Path("./src/weights/results")
MODEL_PERFORMANCE_OUTPUT_PATH = Path("./src/temp/performance_history.csv")
MODEL_OUTPUT_PATH = Path("./src/weights")


def retrain_models():
    print("Retraining models")
    try:
        data = pd.read_parquet(CURATED_DATA_PATH)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Stratify split to ensure equal distribution of data
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data.type
    )

    # Vectorize text data
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", tokenizer=Lemmatizer()
    )
    X_train = vectorizer.fit_transform(train_data.posts)
    X_test = vectorizer.transform(test_data.posts)

    # Save the vectorizer as a joblib file
    try:
        joblib.dump(vectorizer, MODEL_OUTPUT_PATH.joinpath("vectorizer.joblib"))
    except Exception as e:
        print(f"Error saving vectorizer as joblib: {e}")

    # Transform target labels to numerical values
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train_data.type)
    y_test = target_encoder.transform(test_data.type)

    # Save the target encoder as a joblib file
    try:
        joblib.dump(target_encoder, MODEL_OUTPUT_PATH.joinpath("target_encoder.joblib"))
    except Exception as e:
        print(f"Error saving target encoder as joblib: {e}")

    # thread here
    thread_log = Thread(
        target=train_and_test_LR,
        args=(X_train, y_train, X_test, y_test),
    )
    thread_lsvc = Thread(
        target=train_and_test_LSVC,
        args=(X_train, y_train, X_test, y_test),
    )
    thread_log.start()
    thread_lsvc.start()
    thread_log.join()
    thread_lsvc.join()


def train_and_test_LR(X_train, y_train, X_test, y_test):
    print("Training and testing Logistic Regression model")

    # Train Logistic Regression model
    model_log = LogisticRegression(max_iter=3000, C=0.5, n_jobs=-1)
    model_log.fit(X_train, y_train)

    # Save the model as a joblib file
    try:
        joblib.dump(
            model_log, MODEL_OUTPUT_PATH.joinpath("logistic_regression_model.joblib")
        )
    except Exception as e:
        print(f"Error saving Logistic Regression model as joblib: {e}")

    # Make predictions on the test set
    test_predictions = model_log.predict(X_test)

    # Combine features, true labels, and predicted labels
    results_df = pd.DataFrame(
        {"True_Labels": y_test, "Predicted_Labels": test_predictions}
    )

    # Calculate model performance and save it
    performance = (results_df.True_Labels == results_df.Predicted_Labels).sum() / len(
        results_df.True_Labels
    )
    with open(MODEL_PERFORMANCE_OUTPUT_PATH, "a") as file:
        file.write(f"{datetime.now().isoformat()},logistic_regression,{performance}\n")

    results_df.to_parquet(
        MODEL_RESULT_OUTPUT_PATH.joinpath("logistic_regression_results.parquet"),
        index=False,
    )

    print(
        f"Test predictions for Logistic Regression model saved at: {MODEL_RESULT_OUTPUT_PATH.joinpath('logistic_regression_results.parquet')}"
    )


def train_and_test_LSVC(X_train, y_train, X_test, y_test):
    print("Training and testing LSVC model")

    # Train LinearSVC model
    model_linear_svc = LinearSVC(C=0.1)
    model_linear_svc.fit(X_train, y_train)

    # Save the model as a joblib file
    try:
        joblib.dump(
            model_linear_svc, MODEL_OUTPUT_PATH.joinpath("linear_svc_model.joblib")
        )
    except Exception as e:
        print(f"Error saving LinearSVC model as joblib: {e}")

    # Make predictions on the test set
    test_predictions = model_linear_svc.predict(X_test)

    # Combine features, true labels, and predicted labels
    results_df = pd.DataFrame(
        {"True_Labels": y_test, "Predicted_Labels": test_predictions}
    )
    results_df.to_parquet(
        MODEL_RESULT_OUTPUT_PATH.joinpath("linearSVC_results.parquet"),
        index=False,
    )

    # Calculate model performance and save it
    performance = (results_df.True_Labels == results_df.Predicted_Labels).sum() / len(
        results_df.True_Labels
    )
    with open(MODEL_PERFORMANCE_OUTPUT_PATH, "a") as file:
        file.write(f"{datetime.now().isoformat()},linear_svc,{performance}\n")

    print(
        f"Test predictions for LinearSVC model saved at: {MODEL_RESULT_OUTPUT_PATH.joinpath('linearSVC_results.parquet')}"
    )
