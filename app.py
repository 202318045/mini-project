from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import librosa

app = FastAPI()

# Load the pre-trained model
MODEL_PATH = r"C:\Users\Tarun\Mini\my-model.h5"
model = load_model(MODEL_PATH)

# Constants for audio preprocessing
SAMPLE_RATE = 16000
DURATION = 2
N_MELS = 128
MAX_TIME_STEPS = 87

# Function for audio preprocessing
def preprocess_audio(audio):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]
    return mel_spectrogram[np.newaxis, ..., np.newaxis]

# Endpoint for model prediction
@app.post("/predict/")
async def predict(audio_data: List[float]):
    try:
        audio = np.array(audio_data)
        mel_spectrogram = preprocess_audio(audio)
        prediction = model.predict(mel_spectrogram)
        predicted_class = np.argmax(prediction)
        return JSONResponse(content={"predicted_class": int(predicted_class)})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error making prediction")
