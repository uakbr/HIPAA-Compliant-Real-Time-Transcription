# Configuration settings (e.g., audio buffer size, PHI detection rules)

class Config:
    # Audio settings
    SAMPLE_RATE = 16000  # 16kHz sample rate
    BUFFER_SECONDS = 30  # Circular buffer length in seconds
    CHUNK_SIZE = int(SAMPLE_RATE * 5)  # 5-second chunks
    FRAMES_PER_BUFFER = 1024

    # Dynamic batching settings
    INITIAL_CHUNK_SIZE = CHUNK_SIZE
    MIN_CHUNK_SIZE = int(SAMPLE_RATE * 5)
    MAX_CHUNK_SIZE = int(SAMPLE_RATE * 30)
    CHUNK_ADJUST_STEP = int(SAMPLE_RATE)
    TARGET_LATENCY = 0.5  # Target latency in seconds

    # PHI detection settings
    PHI_ENTITY_LABELS = ['PERSON', 'DATE', 'ORG', 'GPE', 'LOC', 'ID']

    # CPU optimization settings
    CPU_NUM_THREADS = 4  # Adjust based on the number of CPU cores

    # Other configurations as needed