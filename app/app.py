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
    
@app.route('/monitoring', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        saved_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_file_path)
        # Process the file
        output_html_path = pre_processed_csv(Path(saved_file_path))
        return send_file(output_html_path, mimetype='text/html')
    else:
        return 'Invalid file type'


if __name__ == '__main__':
    app.run(debug=True)
