# Dynamic batching logic for adjusting audio chunk size in real-time

from utils.config import Config

class DynamicBatcher:
    def __init__(self):
        self.config = Config()
        self.current_chunk_size = self.config.INITIAL_CHUNK_SIZE
        self.min_chunk_size = self.config.MIN_CHUNK_SIZE
        self.max_chunk_size = self.config.MAX_CHUNK_SIZE

    def adjust_chunk_size(self, processing_latency):
        # Adjust chunk size based on processing latency
        target_latency = self.config.TARGET_LATENCY
        if processing_latency > target_latency:
            new_chunk_size = max(self.min_chunk_size, self.current_chunk_size - self.config.CHUNK_ADJUST_STEP)
        elif processing_latency < target_latency * 0.8:
            new_chunk_size = min(self.max_chunk_size, self.current_chunk_size + self.config.CHUNK_ADJUST_STEP)
        else:
            new_chunk_size = self.current_chunk_size

        if new_chunk_size != self.current_chunk_size:
            self.current_chunk_size = new_chunk_size
            print(f"Adjusted chunk size to {self.current_chunk_size / self.config.SAMPLE_RATE:.2f} seconds.")

    def get_current_chunk_size(self):
        return self.current_chunk_size