# Unit tests for model optimizations

import unittest
from src.whisper_model import WhisperModel
import numpy as np

class TestWhisperModelOptimizations(unittest.TestCase):
    def test_mixed_precision_inference(self):
        model = WhisperModel(use_gpu=True)
        audio_data = np.random.randn(16000).astype(np.float32)
        transcription = model.transcribe(audio_data)
        self.assertIsInstance(transcription, str)

    def test_quantized_cpu_inference(self):
        model = WhisperModel(use_gpu=False)
        audio_data = np.random.randn(16000).astype(np.float32)
        transcription = model.transcribe(audio_data)
        self.assertIsInstance(transcription, str)

if __name__ == '__main__':
    unittest.main()