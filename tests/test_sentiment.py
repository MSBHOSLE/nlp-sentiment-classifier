import unittest
from nlp_model.classifier import SentimentClassifier


class TestSentimentClassifier(unittest.TestCase):
    def test_positive_classification(self):
        text = "This application is amazing, super fast and reliable!"
        res = SentimentClassifier.predict(text)
        self.assertEqual(res["label"], "POSITIVE")
        self.assertGreaterEqual(res["confidence"], 0.7)

    def test_negative_classification(self):
        text = "The system is broken, slow and terrible to use."
        res = SentimentClassifier.predict(text)
        self.assertEqual(res["label"], "NEGATIVE")
        self.assertGreaterEqual(res["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
