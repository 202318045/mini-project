import streamlit as st
import requests
import numpy as np
import librosa

# Define constants
API_URL = "http://127.0.0.1:8000"  # Change to the appropriate API URL
SAMPLE_RATE = 16000
DURATION = 2

# Function to preprocess audio files
def preprocess_audio(audio_file):
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
    audio = audio.tolist()  # Convert to list for JSON serialization
    return audio

# Function to make prediction
def predict(audio_data):
    response = requests.post(f"{API_URL}/predict/", json=audio_data)
    if response.status_code == 200:
        predicted_class = response.json()["predicted_class"]
        return predicted_class
    else:
        st.error("Error making prediction")

# Streamlit UI
st.title("Audio Classification Demo")

uploaded_file = st.file_uploader("Upload an audio file (.wav)")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict"):
        st.write("Preprocessing audio...")
        audio_data = preprocess_audio(uploaded_file)
        predicted_class = predict(audio_data)
        st.write(f"Predicted class: {predicted_class}")
