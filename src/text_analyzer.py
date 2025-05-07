import logging
import os
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import gensim
import gensim.downloader as api
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine

# Set up logger for this module
logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    A class for analyzing text data in DataFrames.

    This class provides methods for:
    - Finding the most common words with specific criteria
    - Calculating similarity between words using word embeddings
    - Extracting and summing dollar amounts from text
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the TextAnalyzer.

        Args:
            model_dir: Directory to store word embedding models (default=None)
                       If None, uses 'models' directory in the project root
        """
        self.word_vectors = None

        # Set model directory path
        if model_dir is None:
            # Get the path of the file where this class is defined
            src_dir = os.path.dirname(os.path.abspath(__file__))
            # Move one level up to the project directory
            project_dir = os.path.dirname(src_dir)
            # Define model path inside project/models/
            self.model_dir = os.path.join(project_dir, "models")
        else:
            self.model_dir = model_dir

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Define unvoiced consonants list for word filtering
        self.unvoiced_consonants = [
            "b",
            "c",
            "d",
            "f",
            "g",
            "h",
            "j",
            "k",
            "l",
            "m",
            "n",
            "p",
            "q",
            "r",
            "s",
            "t",
            "v",
            "w",
            "x",
            "z",
        ]

    def _load_word_vectors(self, model_name: str = "glove-wiki-gigaword-50") -> None:
        """
        Load word vectors model from disk or download if not available.

        Args:
            model_name: Name of the model to load from gensim
        """
        model_path = os.path.join(self.model_dir, "word2vec.model")

        if os.path.exists(model_path):
            logger.info("Loading existing word embedding model...")
            self.word_vectors = gensim.models.KeyedVectors.load(model_path)
        else:
            logger.info(f"Downloading and saving {model_name} model...")
            self.word_vectors = api.load(model_name)
            self.word_vectors.save(model_path)

    def _validate_dataframe(self, df: pd.DataFrame, text_column: str) -> None:
        """
        Validate that the DataFrame contains the required text column.

        Args:
            df: DataFrame to validate
            text_column: Name of the text column that should exist

        Raises:
            ValueError: If the text column doesn't exist in the DataFrame
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")

    def _extract_words(self, text: str, min_length: int = 1) -> List[str]:
        """
        Extract words from text that meet the minimum length requirement.

        Args:
            text: Text string to process
            min_length: Minimum length of words to include

        Returns:
            List of lowercase words that meet the criteria
        """
        # Convert to lowercase and extract words
        return [
            word
            for word in re.findall(r"\b[a-z]+\b", text.lower())
            if len(word) >= min_length
        ]

    def get_top_words(
        self,
        df: pd.DataFrame,
        text_column: str,
        min_length: int = 5,
        top_n: int = 10,
        gender_filter: Optional[str] = "female",
        age_min: Optional[int] = 20,
        age_max: Optional[int] = 30,
        gender_column: str = "gender",
        age_column: str = "age",
        consonant_filter: bool = True,
    ) -> List[Tuple[str, int]]:
        """
        Extract the top N most common words that meet specific criteria from a DataFrame.

        Args:
            df: DataFrame containing the text data
            text_column: Name of column containing text to analyze
            min_length: Minimum length of words to consider (default=5)
            top_n: Number of top words to return (default=10)
            gender_filter: Filter by gender value if specified (default=None)
            age_min: Minimum age to include (inclusive, default=None)
            age_max: Maximum age to include (inclusive, default=None)
            gender_column: Name of column containing gender data (default='gender')
            age_column: Name of column containing age data (default='age')
            consonant_filter: If True, only include words that start and end with unvoiced consonants

        Returns:
            List of tuples containing (word, frequency) sorted by frequency
        """
        logger.info("Getting top words started")

        # Validate input DataFrame
        self._validate_dataframe(df, text_column)

        # Apply filters if specified
        filtered_df = df.copy()
        if gender_filter is not None and gender_column in df.columns:
            filtered_df = filtered_df[filtered_df[gender_column] == gender_filter]
        if age_min is not None and age_column in df.columns:
            filtered_df = filtered_df[filtered_df[age_column] >= age_min]
        if age_max is not None and age_column in df.columns:
            filtered_df = filtered_df[filtered_df[age_column] <= age_max]

        # Process all text
        all_words = []
        for text in filtered_df[text_column].dropna():
            words = self._extract_words(text, min_length)

            # Apply additional filtering if needed
            if consonant_filter:
                words = [
                    word
                    for word in words
                    if (
                        word[0] in self.unvoiced_consonants
                        and word[-1] in self.unvoiced_consonants
                    )
                ]

            all_words.extend(words)

        # Count word frequencies
        word_counts = Counter(all_words)

        logger.info("Get top words ran successfully")
        # Return top N words
        return word_counts.most_common(top_n)

    def get_top_similar_words(
        self,
        df: pd.DataFrame,
        text_column: str,
        min_length: int = 6,
        top_n: int = 100,
        similarity_limit: int = 50,
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of similar words in the DataFrame based on word embeddings.

        Args:
            df: DataFrame containing the text data
            text_column: Name of column containing text to analyze
            min_length: Minimum length of words to consider (default=5)
            top_n: Number of top words to use for similarity (default=8000)
            similarity_limit: Number of similar word pairs to return (default=10)

        Returns:
            List of tuples containing (word1, word2, similarity) sorted by similarity
        """
        logger.info("Get top similar words started")

        # Validate input DataFrame
        self._validate_dataframe(df, text_column)

        # Load word vectors if not already loaded
        if self.word_vectors is None:
            self._load_word_vectors()

        # Process all text
        logger.info(f"Minimal length of words: {min_length}")
        all_words = []

        for text in df[text_column].dropna():
            words = self._extract_words(text, min_length)
            all_words.extend(words)

        # Count word frequencies
        word_counts = Counter(all_words)

        # Get top N words to calculate similarity on
        common_words = [word for word, _ in word_counts.most_common(top_n)]
        logger.info(f"Number of unique words: {len(common_words)}")

        # Define function to compute similarity
        def compute_similarity(w1, w2):
            if w1 in self.word_vectors and w2 in self.word_vectors:
                return (
                    w1,
                    w2,
                    1 - cosine(self.word_vectors[w1], self.word_vectors[w2]),
                )
            return None

        # Calculate similarities in parallel
        results = Parallel(n_jobs=-1)(
            delayed(compute_similarity)(w1, w2)
            for w1, w2 in combinations(common_words, 2)
        )

        # Remove None values and sort by similarity
        similarities = sorted(filter(None, results), key=lambda x: -x[2])[
            :similarity_limit
        ]

        logger.info("Get top similar words ran successfully")
        return similarities

    def sum_dollar_amounts(self, df: pd.DataFrame, text_column: str) -> float:
        """
        Extract all dollar amounts from text and return their sum.

        Args:
            df: DataFrame containing the text data
            text_column: Name of column containing text to analyze

        Returns:
            float: Sum of all dollar amounts found in the text
        """
        logger.info("Sum dollar amounts started")

        # Validate input DataFrame
        self._validate_dataframe(df, text_column)

        # Define dollar amount patterns
        dollar_patterns = [
            r"\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)",  # $10, $10.50, $1,000.25
            r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*dollars",  # 10 dollars, 10.50 dollars
            r"(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*USD",  # 10 USD, 10.50 USD
        ]

        # Process all text
        total_amount = 0.0
        for text in df[text_column].dropna():
            for pattern in dollar_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Remove commas from numbers like 1,000.25
                    amount_str = match.replace(",", "")
                    try:
                        amount = float(amount_str)
                        total_amount += amount
                    except ValueError:
                        logger.warning(f"Could not convert '{match}' to float")

        logger.info(f"Sum dollar amounts completed. Total: ${total_amount:.2f}")
        return total_amount
