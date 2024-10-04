# Unit tests for Whisper model wrapper and inference

import unittest
from src.whisper_model import WhisperModel
import numpy as np

class TestWhisperModel(unittest.TestCase):
    def setUp(self):
        # Initialize the model (use a small model or mock for testing)
        self.model = WhisperModel(use_gpu=False)

    def test_transcribe_empty_audio(self):
        audio_data = np.array([], dtype=np.int16)
        transcription = self.model.transcribe(audio_data)
        self.assertEqual(transcription.strip(), "")

    def test_transcribe_silent_audio(self):
        # Generate silent audio (zeros)
        audio_data = np.zeros(16000, dtype=np.int16)
        transcription = self.model.transcribe(audio_data)
        # Expect empty or silence indication
        self.assertIsInstance(transcription, str)

    def test_preprocess_audio(self):
        # Test normalization of audio data
        audio_data = np.array([0, -32768, 32767], dtype=np.int16)
        audio_tensor = self.model.preprocess_audio(audio_data)
        self.assertEqual(audio_tensor.shape[0], 3)
        self.assertAlmostEqual(audio_tensor[0].item(), 0.0)
        self.assertAlmostEqual(audio_tensor[1].item(), -1.0)
        self.assertAlmostEqual(audio_tensor[2].item(), 1.0)

if __name__ == '__main__':
    unittest.main()