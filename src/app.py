from monitoring import monitor_directory
from lemmatizer import Lemmatizer as Lemmatizer

from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from threading import Thread
from pathlib import Path
import numpy as np

import warnings
import joblib
import nltk
import os

# create the data directories
RAW_DATA_PATH = Path("./src/data/raw")
PRE_CURATED_DATA_DIR = Path("./src/data/pre_curated")
CURATED_DATA_DIR = Path("./src/data/curated")
os.makedirs("./src/data", exist_ok=True)
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PRE_CURATED_DATA_DIR, exist_ok=True)
os.makedirs(CURATED_DATA_DIR, exist_ok=True)

# create the temp directorie
PROCESSED_FILE_HASHES_FILE = Path("./src/temp/processed_file_hashes.txt")
os.makedirs("./src/temp", exist_ok=True)

# create the weights directory
MODEL_RESULT_OUTPUT_PATH = Path("./src/weights/results")
os.makedirs(MODEL_RESULT_OUTPUT_PATH, exist_ok=True)

# download the necessary nltk resources
nltk.download("wordnet")

# ignore warnings
warnings.filterwarnings("ignore")

# create the Flask app
app = Flask(__name__)

# load the trained model and necessary preprocessing objects
vectorizer = joblib.load("./src/weights/vectorizer.joblib")
target_encoder = joblib.load("./src/weights/target_encoder.joblib")
model_log = joblib.load("./src/weights/logistic_regression_model.joblib")

# unused for now
# model_lsvc = joblib.load("./src/weights/linear_svc_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    post = request.form["post"]
    # Prédire les probabilités pour chaque classe
    probabilities = model_log.predict_proba(vectorizer.transform([post]).toarray())[0]
    
    # Obtenir les indices des trois prédictions les plus élevées
    top_predictions_indices = np.argsort(-probabilities)[:3]
    top_predictions = target_encoder.inverse_transform(top_predictions_indices)
    top_probabilities = probabilities[top_predictions_indices]
    
    # Préparer le chemin de l'image pour la prédiction la plus élevée
    top_prediction = top_predictions[0]
    image_filename = f"mbti_images/{top_prediction}.png"
    
    # Préparer les données pour le template
    predictions_data = zip(top_predictions, top_probabilities)
    
    return render_template(
        "result.html",
        post=post,
        predictions_data=predictions_data,
        image_filename=image_filename,
        top_prediction=top_prediction
    )


@app.route("/monitoring", methods=["GET"])
def monitoring():
    return render_template("monitoring.html")


@app.route("/monitoring", methods=["POST"])
def upload_csv_for_monitoring():
    # check if the post request has the file part
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    # check if the file is a CSV
    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        file_path = os.path.join(RAW_DATA_PATH, filename)
        file.save(file_path)
        print(f"File {filename} uploaded successfully")

        # redirect to the monitoring page
        return redirect(url_for("monitoring"))
    else:
        return "Invalid file type, please upload a CSV."


if __name__ == "__main__":
    # launch monitoring in a thread
    monitoring_thread = Thread(target=monitor_directory)
    monitoring_thread.daemon = True

    try:
        monitoring_thread.start()
        app.run(debug=False)
    except KeyboardInterrupt:
        monitoring_thread.join()
