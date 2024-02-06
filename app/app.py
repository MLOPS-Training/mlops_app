from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import os, sys
import joblib


from data_pipeline.pre_processing.lemmatizer import Lemmatizer
import warnings

from monitoring.monitoring import monitor_directory

warnings.filterwarnings('ignore')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../data_pipeline/data/raw'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

print(f"{sys.path} ICIIIIIII")

# Load the trained model and necessary preprocessing objects
model_log = joblib.load('./joblibs/logistic_regression_model.joblib')
vectorizer = joblib.load('./joblibs/tfidf_vectorizer.joblib')
target_encoder = joblib.load('./joblibs/label_encoder.joblib')


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
def upload_csv_for_monitoring():
    # Vérifiez si le fichier a été téléchargé correctement
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Vérifiez que le fichier est un CSV
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        monitor_directory()
        
        # Redirigez vers une nouvelle page ou retournez un résultat
        return redirect(url_for('monitoring'))
    else:
        return 'Invalid file type, please upload a CSV.'

if __name__ == '__main__':
    app.run(debug=False)
