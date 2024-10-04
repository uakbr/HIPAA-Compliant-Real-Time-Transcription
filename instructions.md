### Technical Software Specification: Real-Time HIPAA-Compliant Transcription using Whisper Model

---

#### Project Overview:
The objective is to develop a desktop application that captures real-time audio through a microphone, transcribes it using the Whisper model hosted locally, and ensures HIPAA compliance by preventing any external data transmission or persistent storage. The application should detect and redact Protected Health Information (PHI) in real time while maintaining transcription speed and accuracy. This solution targets healthcare professionals in clinical settings where data privacy and compliance are critical.

---

### System Components:

1. **Audio Capture and Processing**:
    - **Microphone Input**: The application captures real-time audio using the system's microphone.
    - **Audio Encoding**: The audio is processed into 16-bit PCM, 16kHz format.
    - **Sliding Window Buffer**: A circular buffer stores the most recent 30 seconds of audio for processing.

2. **Local Whisper Model Deployment**:
    - **Whisper Model**: A pre-trained Whisper model from OpenAI is used for transcription. This model is hosted locally using Docker, eliminating any network dependency.
    - **GPU Acceleration**: For real-time processing, the application uses CUDA for GPU acceleration to minimize transcription latency.
    - **Dynamic Batching**: Adaptive batching of audio chunks (5–30 seconds) with overlapping segments ensures continuity of transcription.

3. **Real-Time PHI Scrubbing**:
    - **NER-based PHI Detection**: Custom machine learning model using spaCy for detecting patient names, dates, and other sensitive information.
    - **Rule-Based PHI Detection**: Regex filters for detecting and redacting medical record numbers, social security numbers, etc.
    - **Immediate PHI Redaction**: Redacted information is replaced by placeholders (e.g., [NAME], [DATE]) before display.

4. **Secure Display**:
    - **UI Rendering**: Electron-based UI for displaying transcriptions temporarily.
    - **Timed Clear**: Transcriptions clear automatically after 30 seconds to avoid persistent display of sensitive data.

5. **Memory Management**:
    - **In-Memory Processing**: All processing occurs in RAM to prevent writing to disk.
    - **Secure Memory Erasure**: Sensitive data is immediately overwritten in memory after use.

6. **User Authentication and Access Control**:
    - **2FA**: Users are authenticated using two-factor authentication (TOTP).
    - **Role-Based Access Control**: Different access levels are enforced based on user roles.

---

### Workflow:

1. **Initialization**:
   - Load the Whisper model and associated resources locally using Docker.
   - Establish audio capture from the microphone.
   
2. **Real-Time Processing**:
   - Continuously capture audio using the circular buffer.
   - Feed audio chunks (with 50% overlap) into the Whisper model for transcription.
   - Process transcribed text through the PHI scrubbing pipeline.

3. **Display and Erasure**:
   - Render transcriptions on the UI with PHI redacted.
   - Automatically clear the display after a set time (configurable).
   - Erase all transcribed data from memory using a secure allocation and deallocation routine.

4. **Shutdown**:
   - On exit, securely dispose of any residual data in memory and reset application state.

---

### Key Libraries and Dependencies:

1. **PyTorch**: For running the Whisper model with GPU acceleration.
2. **spaCy**: For Named Entity Recognition (NER) used in PHI detection.
3. **Electron**: For building a secure desktop user interface.
4. **Docker**: For containerizing the Whisper model and its dependencies.
5. **CUDA**: For enabling GPU acceleration.

---

### Performance and Scalability Considerations:

1. **Real-Time Performance**: Achieving real-time transcription on machines equipped with NVIDIA GPUs using CUDA for fast inference.
2. **Fallback for CPU-Only Systems**: The system includes fallback support for CPU-based processing, though with reduced performance.
3. **Dynamic Resource Allocation**: The application dynamically adjusts batching and audio processing based on system load to ensure low latency.
4. **Scalability**: For deployment across larger healthcare facilities, the application can scale by distributing the workload across multiple machines, with each running its own instance of the Whisper model.

---

### Hardware Requirements:

- **GPU**: NVIDIA with at least 4GB VRAM (for optimal performance).
- **CPU**: 4-core processor (for fallback operation).
- **RAM**: Minimum of 16GB.
- **Storage**: SSD (for fast application load times).

Here is the proposed GitHub repository architecture with the files and their corresponding purposes:

```plaintext
whisper-hipaa-transcription/
├── Dockerfile                         # Dockerfile for containerizing the Whisper model with CUDA support
├── README.md                          # Project overview, installation instructions, and usage details
├── requirements.txt                   # Python dependencies required for the project
├── .gitignore                         # Git ignore file to exclude unnecessary files
├── src/
│   ├── app.py                         # Main entry point for the desktop application, initializes audio capture and UI
│   ├── whisper_model.py               # Wrapper class to handle loading and inference of the Whisper model
│   ├── audio_capture.py               # Module for handling real-time audio capture and encoding
│   ├── phi_scrubber.py                # Logic for detecting and redacting PHI from transcribed text
│   ├── dynamic_batcher.py             # Dynamic batching logic for adjusting audio chunk size in real-time
│   ├── ui/
│   │   ├── electron_app.js            # Electron app initialization and rendering logic for the user interface
│   │   ├── secure_display.js          # Handles secure display and timed clearing of transcriptions
│   │   ├── styles.css                 # UI styling for Electron app
│   │   └── auth_manager.js            # User authentication logic (2FA, role-based access)
│   ├── utils/
│   │   ├── memory_manager.py          # Custom secure memory allocator and erasure functions
│   │   ├── gpu_monitor.py             # Monitors GPU availability and usage for optimization
│   │   ├── logging.py                 # Custom logging for application status (no sensitive data logging)
│   │   └── config.py                  # Configuration settings (e.g., audio buffer size, PHI detection rules)
├── tests/
│   ├── test_audio_capture.py          # Unit tests for audio capture and encoding module
│   ├── test_whisper_model.py          # Unit tests for Whisper model wrapper and inference
│   ├── test_phi_scrubber.py           # Unit tests for PHI detection and scrubbing logic
│   ├── test_ui.py                     # Unit tests for the Electron-based UI
│   └── test_memory_manager.py         # Unit tests for secure memory allocation and deallocation
└── docs/
    ├── architecture.md                # Detailed technical architecture of the project
    ├── api_docs.md                    # API documentation for the model and audio processing components
    └── hipaa_compliance.md            # Explanation of HIPAA compliance features and regulatory adherence
```

### Breakdown of Key Files:

- **Dockerfile**: Sets up the Docker environment with CUDA support for running the Whisper model. It includes all necessary dependencies to isolate the environment from the host system.
  
- **src/app.py**: Main entry point for the application, responsible for initializing audio capture, invoking the Whisper model, processing transcriptions, and launching the Electron UI for display.

- **src/whisper_model.py**: Handles loading the Whisper model locally, including optimizations for GPU acceleration and inference on audio segments.

- **src/audio_capture.py**: Module that manages real-time audio capture using a circular buffer and prepares audio for the Whisper model.

- **src/phi_scrubber.py**: Implements PHI scrubbing logic by using Named Entity Recognition (NER) and regex-based detection to redact sensitive information from transcriptions.

- **src/dynamic_batcher.py**: Contains the logic for adjusting audio segment size dynamically based on system load to ensure low-latency transcription.

- **src/ui/**: Contains the Electron-based front-end files, including:
  - **electron_app.js**: Initializes the Electron app and manages interaction between the UI and the back-end.
  - **secure_display.js**: Handles the secure, temporary display of transcription and automatic text clearing.
  - **auth_manager.js**: Manages user authentication, including multi-factor authentication (MFA) and role-based access control.
  - **styles.css**: Defines the UI styling for the Electron interface.

- **src/utils/**: Contains utility modules:
  - **memory_manager.py**: Implements secure memory management, ensuring sensitive data is immediately zeroed out after use.
  - **gpu_monitor.py**: Monitors GPU usage and optimizes resource allocation.
  - **logging.py**: Custom logging module for the application (ensures no sensitive data is logged).
  - **config.py**: Configuration file that stores application settings such as buffer sizes, audio parameters, and PHI redaction rules.

- **tests/**: Contains unit tests for each of the major modules to ensure functionality, including:
  - Audio capture, Whisper model inference, PHI scrubbing, UI rendering, and memory management.

- **docs/**: Documentation folder:
  - **architecture.md**: Detailed explanation of the system architecture and component interactions.
  - **api_docs.md**: API documentation for developers working with the Whisper model or integrating the application with other systems.
  - **hipaa_compliance.md**: Describes the HIPAA-compliant design features and how the application adheres to regulatory standards.
