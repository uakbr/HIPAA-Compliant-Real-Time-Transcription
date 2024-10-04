# Unit tests for PHI detection and scrubbing logic

import unittest
from src.phi_scrubber import PHIScrubber

class TestPHIScrubber(unittest.TestCase):
    def setUp(self):
        self.scrubber = PHIScrubber()

    def test_scrub_person_name(self):
        text = "Patient John Doe came in for a checkup."
        scrubbed_text = self.scrubber.scrub(text)
        self.assertNotIn("John Doe", scrubbed_text)
        self.assertIn("[PERSON]", scrubbed_text)

    def test_scrub_date(self):
        text = "The appointment is scheduled for July 20, 2022."
        scrubbed_text = self.scrubber.scrub(text)
        self.assertNotIn("July 20, 2022", scrubbed_text)
        self.assertIn("[DATE]", scrubbed_text)

    def test_scrub_ssn(self):
        text = "SSN: 123-45-6789"
        scrubbed_text = self.scrubber.scrub(text)
        self.assertNotIn("123-45-6789", scrubbed_text)
        self.assertIn("[SSN]", scrubbed_text)

    def test_no_phi(self):
        text = "The patient is in good health."
        scrubbed_text = self.scrubber.scrub(text)
        self.assertEqual(text, scrubbed_text)

if __name__ == '__main__':
    unittest.main()