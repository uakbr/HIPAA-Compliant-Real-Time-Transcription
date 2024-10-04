# Module for handling real-time audio capture and encoding

import pyaudio
import threading
import numpy as np
from collections import deque
from utils.config import Config

class AudioCapture:
    def __init__(self):
        self.config = Config()
        self.audio_buffer = deque(maxlen=int(self.config.BUFFER_SECONDS * self.config.SAMPLE_RATE))
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False

    def start(self):
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.config.FRAMES_PER_BUFFER,
            stream_callback=self.callback
        )
        self.is_running = True
        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)

    def get_audio_chunk(self):
        if len(self.audio_buffer) >= self.config.CHUNK_SIZE:
            chunk = np.array([self.audio_buffer.popleft() for _ in range(self.config.CHUNK_SIZE)])
            return chunk
        else:
            return np.array([])

    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.audio_interface.terminate()
            self.is_running = False