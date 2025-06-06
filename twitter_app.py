# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import re

# Load saved model and tokenizer
model = load_model(r"C:\Users\Sariga\OneDrive\Desktop\Twitter_project\sentiment_model.h5")
tokenizer = load(r"C:\Users\Sariga\OneDrive\Desktop\Twitter_project\tokenizer.joblib")

# Constants
MAX_LEN = 100
sentiment_labels = ['Negative', 'Neutral', 'Positive']

# Function to clean and preprocess user input
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    return padded

# Streamlit App UI
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment:")

user_input = st.text_area("Tweet Text", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        predicted_class = np.argmax(prediction)
        sentiment = sentiment_labels[predicted_class]
        st.success(f"Predicted Sentiment: **{sentiment}**")
