import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from src.data_viz import (
    create_gender_histogram,
    create_sign_industry_heatmap,
    create_top_industry_barplot,
    filter_dataframe,
)
from src.logger_config import setup_logger
from src.text_analyzer import TextAnalyzer

# Set up logging system
log_file = setup_logger()

# Get logger for this module
logger = logging.getLogger(__name__)


def main():
    logger.info("Application started")

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate one level up and then into the "data" folder
    data_path = os.path.join(script_dir, "data", "blogtext.csv")

    # Convert to absolute path and read df
    data_path = os.path.abspath(data_path)

    logger.info("Reading data")
    df = pd.read_csv(data_path)
    logger.info("Data succesfully read")

    analyzer = TextAnalyzer()

    # Find top words
    top_words = analyzer.get_top_words(df, "text")
    print("Most common words: ")
    for word in top_words:
        print(word)

    # Find similar words
    similar_words = analyzer.get_top_similar_words(df, "text", min_length=6, top_n=10)
    print("Most similar words:")
    for word in similar_words:
        print(word)

    # Sum dollar amounts
    total_dollars = analyzer.sum_dollar_amounts(df, "text")
    print(f"Total sum of dollars: {total_dollars}")

    # Check if 'plots' directory exists, if not, create it
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Create and save gender histogram
    create_gender_histogram(df, plots_dir)

    # Filter dataframe for topic analysis
    df_filtered = filter_dataframe(df)

    # Create and save top industry barplot
    create_top_industry_barplot(df_filtered, plots_dir)

    # Create sign-industry heatmap
    create_sign_industry_heatmap(df_filtered, plots_dir)

    logger.info("Application finished")


if __name__ == "__main__":
    main()
