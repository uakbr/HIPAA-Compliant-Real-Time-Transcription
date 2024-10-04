# Unit tests for audio capture and encoding module

import unittest
from src.audio_capture import AudioCapture
from unittest.mock import patch, MagicMock

class TestAudioCapture(unittest.TestCase):
    def setUp(self):
        # Patch pyaudio interfaces to prevent actual hardware access
        self.patcher = patch('src.audio_capture.pyaudio.PyAudio')
        self.mock_pyaudio = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.audio_capture = AudioCapture()

    def test_initialization(self):
        self.assertFalse(self.audio_capture.is_running)
        self.assertIsNone(self.audio_capture.stream)

    def test_start_and_stop(self):
        self.audio_capture.start()
        self.assertTrue(self.audio_capture.is_running)
        self.audio_capture.stop()
        self.assertFalse(self.audio_capture.is_running)

    def test_get_audio_chunk(self):
        # Mock audio data
        self.audio_capture.audio_buffer.extend([0]*self.audio_capture.config.CHUNK_SIZE)
        chunk = self.audio_capture.get_audio_chunk()
        self.assertEqual(len(chunk), self.audio_capture.config.CHUNK_SIZE)

if __name__ == '__main__':
    unittest.main()