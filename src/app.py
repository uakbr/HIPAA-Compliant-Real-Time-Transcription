# Main entry point for the desktop application, initializes audio capture and UI

import threading
import time
import signal
import sys
from audio_capture import AudioCapture
from whisper_model import WhisperModel
from phi_scrubber import PHIScrubber
from ui.electron_app import ElectronApp
from utils.gpu_monitor import GPUMonitor
from utils.memory_manager import SecureAllocator
from dynamic_batcher import DynamicBatcher
from utils.logging import AppLogger

def main():
    # Initialize secure allocator
    allocator = SecureAllocator()
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    gpu_available = gpu_monitor.is_gpu_available()

    # Initialize Whisper model
    whisper_model = WhisperModel(use_gpu=gpu_available)

    # Initialize PHI scrubber
    phi_scrubber = PHIScrubber()

    # Initialize audio capture
    audio_capture = AudioCapture()
    audio_capture.start()

    # Initialize UI
    ui_app = ElectronApp()

    # Initialize dynamic batcher
    dynamic_batcher = DynamicBatcher()

    # Initialize logger
    logger = AppLogger()

    # Function to process audio and update UI
    def process_audio():
        while True:
            start_time = time.time()

            chunk_size = dynamic_batcher.get_current_chunk_size()
            audio_chunk = audio_capture.get_audio_chunk(chunk_size)
            if audio_chunk.size == 0:
                continue

            # Securely allocate audio data
            audio_chunk = allocator.allocate_buffer(audio_chunk)

            transcription = whisper_model.transcribe(audio_chunk)
            sanitized_text = phi_scrubber.scrub(transcription)
            ui_app.display_transcription(sanitized_text)

            # Measure processing latency
            processing_latency = time.time() - start_time
            dynamic_batcher.adjust_chunk_size(processing_latency)

            # Log processing time
            logger.info(f"Processed {chunk_size} samples in {processing_latency:.2f} seconds.")

            # Securely erase used data
            allocator.deallocate(audio_chunk)
            allocator.deallocate(transcription)
            allocator.deallocate(sanitized_text)

    # Function to handle secure shutdown
    def secure_shutdown():
        logger.info("Performing secure shutdown...")
        # Stop audio capture
        audio_capture.stop()
        # Securely erase data
        allocator.cleanup()
        # Cleanup Whisper model
        whisper_model.cleanup()
        # Shutdown UI
        ui_app.quit()
        logger.info("Shutdown complete.")
        sys.exit(0)

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: secure_shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: secure_shutdown())

    # Start audio processing thread
    processing_thread = threading.Thread(target=process_audio)
    processing_thread.start()

    try:
        # Run the UI application
        ui_app.run()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        secure_shutdown()

    # On exit, perform cleanup
    audio_capture.stop()
    processing_thread.join()
    whisper_model.cleanup()
    allocator.cleanup()

if __name__ == "__main__":
    main()