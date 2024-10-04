# Custom logging for application status (no sensitive data logging)

import logging
import re

class AppLogger:
    def __init__(self, log_file='application.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger()

    def info(self, message):
        self.logger.info(self.sanitize_message(message))

    def warning(self, message):
        self.logger.warning(self.sanitize_message(message))

    def error(self, message):
        self.logger.error(self.sanitize_message(message))

    def sanitize_message(self, message):
        # Remove any sensitive information using regex
        sanitized_message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', message)
        # Add more sanitization rules as needed
        return sanitized_message