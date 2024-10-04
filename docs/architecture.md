# Detailed Technical Architecture

## Overview

The application is designed with a modular architecture to ensure scalability, maintainability, and security. Each component is responsible for a specific functionality, and components communicate through well-defined interfaces.

## Components

### 1. Audio Capture and Processing

- **Module**: `audio_capture.py`
- Captures real-time audio input.
- Uses a circular buffer to store audio data.
  
### 2. Whisper Model Integration

- **Module**: `whisper_model.py`
- Loads the Whisper model locally using PyTorch.
- Supports GPU acceleration with CUDA.
  
### 3. PHI Scrubbing

- **Module**: `phi_scrubber.py`
- Utilizes NER and regex patterns to detect and redact PHI.
- Supports custom NER models for enhanced medical entity recognition.

### 4. User Interface

- **Modules**: `electron_app.js`, `auth_manager.js`, `secure_display.js`, `styles.css`
- Provides a secure UI for displaying transcriptions.
- Implements 2FA and RBAC for user authentication and access control.

### 5. Memory Management

- **Module**: `memory_manager.py`
- Provides secure allocation and deallocation of memory.
- Ensures sensitive data is overwritten in memory after use.

### 6. Logging

- **Module**: `logging.py`
- Implements custom logging with message sanitization.
- Prevents logging of sensitive data.

### 7. Dynamic Batching

- **Module**: `dynamic_batcher.py`
- Adjusts audio chunk sizes based on processing latency.

## Data Flow Diagram

[Include a diagram illustrating the flow of data through the system.]

## Security Considerations

- All processing is done locally to maintain data privacy.
- Sensitive data is managed securely in memory and not written to disk.
- User authentication with 2FA and RBAC enhances access security.