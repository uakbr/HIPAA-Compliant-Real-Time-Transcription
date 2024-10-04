#!/bin/bash
# Script to start the application

# Build Docker image if not already built
if [[ "$(docker images -q whisper_model_image:latest 2> /dev/null)" == "" ]]; then
  docker build -t whisper_model_image -f Dockerfile .
fi

# Start Docker container for Whisper model
docker run --gpus all -d --rm --name whisper_model_container whisper_model_image

# Install Node.js dependencies
cd src/ui
npm install

# Start the Electron UI
npm run start &

# Start the Python application
cd ../..
python3 src/app.py