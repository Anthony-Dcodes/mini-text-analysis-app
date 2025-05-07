import os
import shutil
import tempfile
import unittest

import pandas as pd

from src.text_analyzer import TextAnalyzer


class TestTextAnalyzer(unittest.TestCase):
    """
    Unit tests for the TextAnalyzer class.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test models/results
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create an instance of TextAnalyzer
        self.analyzer = TextAnalyzer(model_dir=self.model_dir)

        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame(
            {
                "text": [
                    "The quick brown fox jumps over the lazy dog. It costs $50.75.",
                    "Hello world! This is a sample text that costs 25 dollars.",
                    "Another example with 100 USD and $25.50 mentioned.",
                ],
                "gender": ["male", "female", "female"],
                "age": [25, 35, 28],
            }
        )

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_extract_words(self):
        """Test the _extract_words method."""
        text = "Hello, world! This is a TEST."
        result = self.analyzer._extract_words(text, min_length=3)

        self.assertIn("hello", result)
        self.assertIn("world", result)
        self.assertIn("this", result)
        self.assertIn("test", result)
        self.assertNotIn("is", result)  # too short
        self.assertNotIn("a", result)  # too short

    def test_validate_dataframe(self):
        """Test the _validate_dataframe method."""
        # Should not raise an exception
        self.analyzer._validate_dataframe(self.sample_data, "text")

        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.analyzer._validate_dataframe(self.sample_data, "non_existent_column")

    def test_sum_dollar_amounts(self):
        """Test the sum_dollar_amounts method."""
        total_amount = self.analyzer.sum_dollar_amounts(self.sample_data, "text")
        # Should find $50.75, 25 dollars, 100 USD and $25.50 for a total of 201.25
        self.assertEqual(total_amount, 201.25)

    def test_get_top_words(self):
        """Test the get_top_words method."""
        # Create a DataFrame with more predictable word content
        test_df = pd.DataFrame(
            {
                "text": [
                    "buffalo buffalo buffalo buffalo buffalo",
                    "buffalo buffalo buffalo buffalo",
                    "test test test test",
                ],
                "gender": ["female", "female", "male"],
                "age": [25, 22, 30],
            }
        )

        # Test with no filtering
        result = self.analyzer.get_top_words(
            test_df,
            "text",
            min_length=4,  # to include 'test'
            consonant_filter=False,  # turn off consonant filtering
            gender_filter=None,
            age_min=None,
            age_max=None,
        )

        # The most common word should be "buffalo"
        self.assertEqual(result[0][0], "buffalo")
        self.assertEqual(result[0][1], 9)  # 9 occurrences

        # Test with gender filtering
        female_result = self.analyzer.get_top_words(
            test_df,
            "text",
            min_length=4,
            consonant_filter=False,
            gender_filter="female",
            age_min=None,
            age_max=None,
        )

        # For females, there should only be "buffalo" words
        self.assertEqual(female_result[0][0], "buffalo")
        self.assertEqual(female_result[0][1], 9)  # 9 occurrences
        self.assertEqual(len(female_result), 1)  # only one word type


if __name__ == "__main__":
    unittest.main()
