# Dynamic batching logic for adjusting audio chunk size in real-time

from utils.config import Config

class DynamicBatcher:
    def __init__(self):
        self.config = Config()
        self.current_chunk_size = self.config.INITIAL_CHUNK_SIZE
        self.min_chunk_size = self.config.MIN_CHUNK_SIZE
        self.max_chunk_size = self.config.MAX_CHUNK_SIZE

    def adjust_chunk_size(self, processing_latency):
        if processing_latency > self.config.TARGET_LATENCY:
            self.current_chunk_size = max(self.min_chunk_size, self.current_chunk_size - self.config.CHUNK_ADJUST_STEP)
        elif processing_latency < self.config.TARGET_LATENCY * 0.8:
            self.current_chunk_size = min(self.max_chunk_size, self.current_chunk_size + self.config.CHUNK_ADJUST_STEP)

    def get_current_chunk_size(self):
        return self.current_chunk_size