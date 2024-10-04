# Main entry point for the desktop application, initializes audio capture and UI

import threading
from audio_capture import AudioCapture
from whisper_model import WhisperModel
from phi_scrubber import PHIScrubber
from ui.electron_app import ElectronApp
from utils.gpu_monitor import GPUMonitor
from utils.memory_manager import SecureAllocator

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

    # Function to process audio and update UI
    def process_audio():
        while True:
            audio_chunk = audio_capture.get_audio_chunk()
            if audio_chunk.size == 0:
                continue

            transcription = whisper_model.transcribe(audio_chunk)
            sanitized_text = phi_scrubber.scrub(transcription)
            ui_app.display_transcription(sanitized_text)

            # Securely erase used data
            allocator.deallocate(audio_chunk)
            allocator.deallocate(transcription)
            allocator.deallocate(sanitized_text)

    # Start audio processing thread
    processing_thread = threading.Thread(target=process_audio)
    processing_thread.start()

    # Run the UI application
    ui_app.run()

    # On exit, perform cleanup
    audio_capture.stop()
    processing_thread.join()
    whisper_model.cleanup()
    allocator.cleanup()

if __name__ == "__main__":
    main()