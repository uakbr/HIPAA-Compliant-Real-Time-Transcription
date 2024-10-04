# Logic for detecting and redacting PHI from transcribed text

import spacy
import re
from utils.config import Config

class PHIScrubber:
    def __init__(self):
        self.config = Config()
        # Load custom NER model for medical entities if available
        try:
            self.nlp = spacy.load('en_core_sci_md')
        except OSError:
            self.nlp = spacy.load('en_core_web_sm')
            print("Custom medical NER model not found. Using default model.")

        # Define regex patterns for rule-based PHI detection
        self.regex_patterns = [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),        # Social Security Number
            (re.compile(r'\b\d{10}\b'), '[PHONE]'),                 # Phone Number
            (re.compile(r'\b\d{5}(-\d{4})?\b'), '[ZIP]'),           # ZIP Code
            (re.compile(r'\b[A-Z]{2}\d{6}\b'), '[MRN]'),            # Medical Record Number
            (re.compile(r'\b\d{4}-\d{2}-\d{2}\b'), '[DATE]'),       # Date of Birth
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