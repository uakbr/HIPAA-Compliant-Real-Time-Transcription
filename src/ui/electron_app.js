// src/ui/electron_app.js
// Electron app initialization and rendering logic for the user interface

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const AuthManager = require('./auth_manager');

let mainWindow;
let authManager = new AuthManager();

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
    },
  });

  // Load the initial HTML file
  mainWindow.loadFile('index.html');

  // Handle authentication
  mainWindow.webContents.on('did-finish-load', () => {
    authManager.authenticate().then((authenticated) => {
      if (authenticated) {
        mainWindow.webContents.send('authenticated', authManager.getCurrentUserRole());
      } else {
        app.quit();
      }
    });
  });

  // Security measures
  mainWindow.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
    return callback(false);
  });

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});