# src/data_viz.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def filter_dataframe(df):
    """
    Filter the dataframe to remove 'indUnk' and 'Student' from the topic column.

    Args:
        df (pandas.DataFrame): Input dataframe with a 'topic' column

    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    return df.loc[(df["topic"] != "indUnk") & (df["topic"] != "Student"), :]


def calculate_cramers_v(contingency_table):
    """
    Calculate Cramer's V statistic for categorical correlation.

    Args:
        contingency_table (pandas.DataFrame): A cross-tabulation of two categorical variables

    Returns:
        tuple: (Cramer's V statistic (between 0 and 1), p-value)
    """
    # Calculate chi-square statistic
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    return cramers_v, p


def create_gender_histogram(df, plots_dir):
    """
    Create and save a histogram of gender distribution.

    Args:
        df (pandas.DataFrame): Dataframe with a 'gender' column
        plots_dir (str): Directory path to save the plot
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df, x="gender", hue="gender", shrink=0.8)
    plt.title("Gender Distribution")

    # Save the plot as a file in the 'plots' directory
    plot_file_path = os.path.join(plots_dir, "gender_histogram.png")
    plt.savefig(plot_file_path)

    # Display the plot
    plt.show()
    plt.close()


def create_top_industry_barplot(df_filtered, plots_dir, n=10):
    """
    Create a bar plot showing the top N topics by count.

    Args:
        df_filtered (pandas.DataFrame): Filtered dataframe with a 'topic' column
        plots_dir (str): Directory path to save the plot
        n (int): Number of top topics to show (default: 10)
    """
    # Get the top N topics and their counts
    top_n = df_filtered["topic"].value_counts().iloc[:n]

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_n.index, y=top_n.values)

    # Improve readability
    plt.title(f"Top {n} Topics by Count")
    plt.xlabel("Industry")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot as a file in the 'plots' directory
    plot_file_path = os.path.join(plots_dir, "top_industry.png")
    plt.savefig(plot_file_path)

    # Display the plot
    plt.show()
    plt.close()


def create_sign_industry_heatmap(df_filtered, plots_dir):
    """
    Create a heatmap showing the distribution of astrology signs within each industry.

    Args:
        df_filtered (pandas.DataFrame): Filtered dataframe with 'topic' and 'sign' columns
        plots_dir (str): Directory path to save the plot
    """
    # Create the contingency table
    contingency_table = pd.crosstab(df_filtered["topic"], df_filtered["sign"])

    # Calculate Cramer's V
    cramers_v, p_value = calculate_cramers_v(contingency_table)

    # Create normalized contingency table
    contingency_normalized = pd.crosstab(
        df_filtered["topic"], df_filtered["sign"], normalize="index"
    )

    # Create the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(contingency_normalized, annot=False, cmap="YlGnBu", linewidths=0.5)

    # Add title and labels
    plt.title(
        f"Distribution of Astrology Signs within Each Industry (Cramer's V: {cramers_v:.3f})"
    )
    plt.xlabel("Astrology sign")
    plt.ylabel("Industry of a blogger")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot as a file in the 'plots' directory
    plot_file_path = os.path.join(plots_dir, "industry_vs_astrology_sign.png")
    plt.savefig(plot_file_path)

    # Display the plot
    plt.show()
    plt.close()
