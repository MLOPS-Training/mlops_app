import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from lemmatizer import Lemmatizer


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

    # Define the directories
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if the last_train_time.txt file exists, and create it if not
    last_train_time_file = models_dir / "last_train_time.txt"
    if not last_train_time_file.exists():
        try:
            # Create the last_train_time.txt file with the current timestamp
            last_train_time_file.write_text(str(curated_data_modified_time))
            print("Created last_train_time.txt with the current timestamp.")
        except Exception as create_error:
            print(f"Error creating last_train_time.txt: {create_error}")

    # Continue with the check for file modification
    try:
        last_train_time = float(last_train_time_file.read_text())
        if curated_data_modified_time <= last_train_time:
            print(
                "No need to re-run training. The curated_data.parquet file hasn't been modified."
            )
            return  # Exit the function if no need to re-run training
    except Exception as read_error:
        print(f"Error reading last_train_time.txt: {read_error}")

    # Stratify split to ensure equal distribution of data (move this here to ensure consistent split)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data.type
    )

    # Train and test LinearSVC
    train_and_test_linear_svc(train_data, test_data)

    # Update the last_train_time file
    last_train_time_file.write_text(str(curated_data_modified_time))


def save_model_and_results(
    model, X_test, y_test, model_name, vectorizer, target_encoder
):
    print(f"Saving {model_name} model and results...")
    # Save the model as a pickle file
    model_output_path_joblib = (
        Path(__file__).resolve().parent
        / "models"
        / "models_train"
        / f"{model_name}_model"
        / "train.joblib"
    )
    model_output_path_joblib.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_output_path_joblib = (
        Path(__file__).resolve().parent
        / "models"
        / "models_train"
        / f"{model_name}_model"
        / "tfidf_vectorizer.joblib"
    )
    vectorizer_output_path_joblib.parent.mkdir(parents=True, exist_ok=True)
    target_encoder_output_path_joblib = (
        Path(__file__).resolve().parent
        / "models"
        / "models_train"
        / f"{model_name}_model"
        / "label_encoder.joblib"
    )
    target_encoder_output_path_joblib.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(model, LinearSVC):
            # Save LinearSVC model using joblib
            joblib.dump(model, model_output_path_joblib)
            print(
                f"{model_name} Model saved successfully as pickle at: {model_output_path_joblib}"
            )
            # Save the vectorizer and target_encoder
            joblib.dump(vectorizer, vectorizer_output_path_joblib)
            print(
                f"Vectorizer saved successfully as joblib at: {vectorizer_output_path_joblib}"
            )
            joblib.dump(target_encoder, target_encoder_output_path_joblib)
            print(
                f"Target Encoder saved successfully as joblib at: {target_encoder_output_path_joblib}"
            )

        else:
            # Handle other models if needed
            pass
        print(
            f"{model_name} Model saved successfully as pickle at: {model_output_path_joblib}"
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
        f"Test predictions for {model_name} saved successfully at: {results_parquet_path}"
    )


def train_and_test_linear_svc(train_data, test_data):
    print("Training and testing LinearSVC model...")
    # Transform target labels to numerical values
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train_data.type)
    y_test = target_encoder.transform(test_data.type)

    # Vectorize text data
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", tokenizer=Lemmatizer()
    )
    X_train = vectorizer.fit_transform(train_data.posts)
    X_test = vectorizer.transform(test_data.posts)

    # Train LinearSVC model
    model_linear_svc = LinearSVC(C=0.1)
    model_linear_svc.fit(X_train, y_train)

    # Save LinearSVC model and results
    save_model_and_results(
        model_linear_svc, X_test, y_test, "linear_svc", vectorizer, target_encoder
    )
