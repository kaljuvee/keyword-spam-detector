# Keyword Spam Detector

This project implements a system to detect keyword spamming in e-commerce product descriptions using both classical machine learning (Random Forest) and LLM-based approaches (OpenAI).

## Overview

Keyword spamming is a practice where sellers include irrelevant keywords in their product descriptions to improve visibility in search results. This creates a poor experience for buyers who see irrelevant products in their search results.

This project provides:
- A Random Forest classifier for detecting keyword spam
- An OpenAI LLM-based detector with few-shot learning
- Comparison of both approaches on a test dataset

## Features

### Random Forest Classifier
- TF-IDF vectorization of text
- Custom feature extraction (hashtag counts, keyword density, etc.)
- Binary classification of spam/not spam

### OpenAI LLM Detector
- Few-shot learning with example spam/not spam descriptions
- Batch processing to handle API limits
- Detailed reasoning for each prediction

## Data

The project uses two datasets:
- `data/train_set.csv`: Training data with product descriptions and spam labels
- `data/test_set.csv`: Test data for evaluation

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- nltk
- spacy
- openai
- python-dotenv

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Download required NLTK and spaCy resources:
   ```
   python -m nltk.downloader punkt stopwords
   python -m spacy download en_core_web_sm
   ```

## Usage

Run the spam detector comparison:

```
python run_spam_detector.py
```

This will:
1. Load the training and test data
2. Train the Random Forest model
3. Evaluate both models on the test data
4. Compare the results

## Results

The script will output performance metrics for both models:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Note: The OpenAI model is evaluated on a subset of the test data to save API costs.

## License

MIT