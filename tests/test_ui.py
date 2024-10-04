# Unit tests for the Electron-based UI

import unittest
import subprocess
import time

class TestUI(unittest.TestCase):
    def test_ui_launch(self):
        # Test if the Electron app launches successfully
        process = subprocess.Popen(['npm', 'run', 'start'], cwd='src/ui')
        time.sleep(5)  # Wait for the app to initialize
        return_code = process.poll()
        self.assertIsNone(return_code)
        process.terminate()
        process.wait()

if __name__ == '__main__':
    unittest.main()