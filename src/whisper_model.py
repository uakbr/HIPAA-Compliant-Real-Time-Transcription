# Wrapper class to handle loading and inference of the Whisper model

import torch
import torchaudio
from utils.config import Config

class WhisperModel:
    def __init__(self, use_gpu=True):
        self.config = Config()
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def load_model(self):
        # Load the Whisper model
        model = torch.hub.load('openai/whisper', 'base')
        model = model.to(self.device)
        model.eval()
        return model

    def transcribe(self, audio_data):
        # Preprocess audio data
        audio_tensor = self.preprocess_audio(audio_data)
        # Perform transcription
        with torch.no_grad():
            result = self.model.transcribe(audio_tensor)
        transcription = result['text']
        return transcription

    def preprocess_audio(self, audio_data):
        # Convert numpy array to tensor
        audio_tensor = torch.from_numpy(audio_data).float() / 32768.0  # Normalize int16 data
        audio_tensor = audio_tensor.to(self.device)
        return audio_tensor

    def cleanup(self):
        # Free up model resources
        del self.model
        torch.cuda.empty_cache()