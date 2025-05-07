from flask import Flask, request, render_template, jsonify
import joblib
import os
import PyPDF2
from docx import Document
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to predict spam or ham
def predict_spam_or_ham(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    score = model.predict_proba(vec)[0][1]
    return pred, score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the user uploaded a file
    file = request.files.get('file')
    message = request.form.get('message')

    if file:
        filename = file.filename
        file_ext = filename.split('.')[-1].lower()

        # Extract text based on file type
        if file_ext == 'pdf':
            text = extract_text_from_pdf(file)
        elif file_ext == 'docx':
            text = extract_text_from_docx(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # If text is extracted, predict spam or ham
        if text.strip():
            pred, score = predict_spam_or_ham(text)
            return jsonify({
                'label': 'Spam' if pred else 'Ham',
                'score': round(score, 2),
                'ham_pct': round((1 - score) * 100, 2),
                'spam_pct': round(score * 100, 2),
                'text': text[:200]  # Returning first 200 characters of the extracted text
            })
        else:
            return jsonify({"error": "No text found in the file"}), 400

    elif message:
        # If the user entered a message
        if message.strip():
            pred, score = predict_spam_or_ham(message)
            return jsonify({
                'label': 'Spam' if pred else 'Ham',
                'score': round(score, 2),
                'ham_pct': round((1 - score) * 100, 2),
                'spam_pct': round(score * 100, 2)
            })
        else:
            return jsonify({"error": "Message cannot be empty"}), 400

    return jsonify({"error": "No message or file provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
