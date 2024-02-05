from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import joblib
from data_pipeline.pre_processing.pre_processing import pre_processed_csv
from lemmatizer import Lemmatizer



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Load the trained model and necessary preprocessing objects
model_log = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
target_encoder = joblib.load('label_encoder.joblib')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/monitoring', methods=['GET'])
def monitoring():
    return render_template('monitoring.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        post = [request.form['post']]
        post_vectorized = vectorizer.transform(post).toarray()
        prediction = target_encoder.inverse_transform(model_log.predict(post_vectorized))[0]
        return render_template('result.html', prediction=prediction)
    


if __name__ == '__main__':
    app.run(debug=True)
