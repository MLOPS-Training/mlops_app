from flask import Flask, render_template, request
import joblib
from lemmatizer import Lemmatizer

app = Flask(__name__)

# Load the trained model and necessary preprocessing objects
model_log = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
target_encoder = joblib.load('label_encoder.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        post = [request.form['post']]
        post_vectorized = vectorizer.transform(post).toarray()
        prediction = target_encoder.inverse_transform(model_log.predict(post_vectorized))[0]
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
