# Logic for detecting and redacting PHI from transcribed text

import spacy
import re
from utils.config import Config

class PHIScrubber:
    def __init__(self):
        self.config = Config()
        self.nlp = spacy.load('en_core_web_sm')
        # Add custom NER model for medical entities if available
        # self.nlp = spacy.load('en_core_med7_lg')

        # Define regex patterns for rule-based PHI detection
        self.regex_patterns = [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
            (re.compile(r'\b\d{10}\b'), '[PHONE]'),
            # Add more patterns as needed
        ]

    def scrub(self, text):
        # Perform NER-based PHI detection
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in self.config.PHI_ENTITY_LABELS:
                text = text.replace(ent.text, f'[{ent.label_}]')

        # Perform rule-based PHI detection
        for pattern, placeholder in self.regex_patterns:
            text = pattern.sub(placeholder, text)

        return text