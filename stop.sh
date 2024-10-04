#!/bin/bash
# Script to stop the application

# Stop the Electron UI
pkill electron

# Stop the Python application
pkill -f src/app.py

# Stop Docker container
docker stop whisper_model_container