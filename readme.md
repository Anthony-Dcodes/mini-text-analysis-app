# Blog Analyzer – A Mini Text Analysis App

## Overview

This project analyzes the Blog Authorship Corpus to extract insights about the content and authors of blog posts. It performs text analysis to identify common words, similar word pairs, and monetary amounts mentioned in the posts, and generates visualizations to illustrate the distribution of blog posts by gender, the top industries mentioned, and the correlation between industry and astrology signs. It was created to showcase some of my python development and data science skills.

## Prerequisites

Before running the project, you need to:

1. **Clone this repository**:

   ```bash
   git clone https://github.com/Anthony-Dcodes/mini-text-analysis-app
   ```

2. **Download the dataset**:

   - Get `blogtext.csv` from [Kaggle](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus/data)
   - Place it inside the `data/` directory

3. **Install dependencies**:

   - Run the following command to create the environment with the required packages:
     ```bash
     conda env create -f environment.yml
     ```
   - Activate the environment:
     ```bash
     conda activate blog_text_env
     ```

## Running the Project

Ensure the blogtext.csv file is in the data folder.

To execute the main analysis, simply run:

```bash
python main.py
```

This script performs the following tasks:

1. **Extracts common words**:

   - Finds the top 10 most common words (minimum 5 letters) used by female bloggers aged 20-30.

2. **Finds similar words**:

   - Identifies 50 pairs of most similar words (at least 6 letters) using cosine similarity with a Word2Vec model.
   - If the model is not found locally, it will be downloaded automatically.

3. **Summarizes dollar amounts**:

   - Calculates the total sum of all dollar amounts mentioned in the blog texts.

4. **Generates charts**:

   - Saves three visualizations in the `plots/` directory:
     - Gender distribution of blog posts (`gender_histogram.png`)
     - Top 10 industries (`top_industry.png`)
     - Correlation between industry and astrology sign (`industry_vs_astrology_sign.png`)

## Running Tests

Basic unit tests for the `TextAnalyzer` class can be run from the project root using:

```bash
python -m unittest discover tests
```

## Project Structure

```
project-root/
│── data/
│   └── blogtext.csv  # Dataset (must be downloaded separately)
│── logs/             # Log files
│── models/           # Word2Vec model (downloaded automatically if missing)
│── notebooks/
│   └── data_exploration.ipynb  # Jupyter notebook for exploration
│── plots/            # Directory for generated charts
│── src/
│   ├── data_viz.py   # Code for generating visualizations
│   ├── logger_config.py  # Logging configuration
│   ├── text_analyzer.py  # Main text analysis module
│── tests/
│   ├── test_text_analyzer.py  # Unit tests for text analysis
│── .gitignore
│── environment.yml  # Environment dependencies
│── main.py          # Entry point for execution
```

## Notes

- The dataset must be manually placed in the `data/` directory.
- Python Version: The project uses Python 3.11, as specified in environment.yml. The Conda environment setup ensures compatibility.
- Disk Space: Ensure sufficient disk space for the blogtext.csv file (approximately 800 MB) and the Word2Vec model (approximately 80 MB).
- The Word2Vec model is not included in the repository but will be automatically downloaded when needed.
- Logs: Check the logs directory for error messages or execution details if issues arise.

## License

This project is licensed under the [MIT License](LICENSE).
