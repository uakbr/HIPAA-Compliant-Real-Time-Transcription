// Handles secure display and timed clearing of transcriptions

const { ipcRenderer } = require('electron');

let transcriptionElement = document.getElementById('transcription');
let clearTimeoutHandle = null;

ipcRenderer.on('new-transcription', (event, text) => {
  displayTranscription(text);
});

function displayTranscription(text) {
  transcriptionElement.textContent = text;

  // Clear existing timeout if any
  if (clearTimeoutHandle) {
    clearTimeout(clearTimeoutHandle);
  }

  // Set timeout to clear the transcription
  clearTimeoutHandle = setTimeout(() => {
    clearTranscription();
  }, 30000); // Clear after 30 seconds
}

function clearTranscription() {
  transcriptionElement.textContent = '';
}