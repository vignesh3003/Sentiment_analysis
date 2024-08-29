from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load the model and vectorizer
with open('sentiment_model_svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer_svm.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return 'Welcome to Emotional Analysis API'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    emotion = prediction[0]
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
ß
