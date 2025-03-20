# Keyword Spam Detector

A tool for detecting keyword spamming in e-commerce product descriptions using Anthropic's Claude LLM via LangChain.

## Overview

This project provides a spam detection system that analyzes product descriptions to identify keyword spamming - the practice of stuffing irrelevant keywords into listings to manipulate search rankings.

Two versions of the detector are available:
1. **Basic version** (`langchain_spam_detector.py`): Uses hardcoded few-shot examples
2. **Enhanced version** (`langchain_spam_detector_v2.py`): Can use examples from your training data

## Requirements

- Python 3.8+
- An Anthropic API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/keyword-spam-detector.git
   cd keyword-spam-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Data Format

The detector expects CSV files with at least two columns:
- `description`: The product description text
- `label`: Binary label (1 for spam, 0 for not spam)

Example files are included in the `data/` directory:
- `data/train_set.csv`: Training data with labeled examples
- `data/test_set.csv`: Test data for evaluation

## Usage

### Original Version

The original detector uses hardcoded few-shot examples:

```bash
# Basic usage (processes 20 examples from test set)
python langchain_spam_detector.py

# Specify a different test file
python langchain_spam_detector.py --test-file path/to/test_data.csv

# Process more examples
python langchain_spam_detector.py --num-examples 50

# Process all examples in the test file
python langchain_spam_detector.py --use-all-data
```

### Enhanced Version (v2)

The enhanced version can use examples from your training data for few-shot learning:

```bash
# Basic usage (same as original)
python langchain_spam_detector_v2.py

# Use training data for few-shot examples
python langchain_spam_detector_v2.py --train-file data/train_set.csv

# Specify number of few-shot examples (balanced between spam/not spam)
python langchain_spam_detector_v2.py --train-file data/train_set.csv --num-few-shot 8

# Force using hardcoded examples even with training data
python langchain_spam_detector_v2.py --train-file data/train_set.csv --use-hardcoded-examples

# Set random seed for reproducible example selection
python langchain_spam_detector_v2.py --train-file data/train_set.csv --random-seed 123
```

## Command Line Arguments

Both versions support these arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--test-file` | Path to test data CSV | `data/test_set.csv` |
| `--train-file` | Path to training data CSV | None |
| `--num-examples` | Number of examples to process | 20 |
| `--use-all-data` | Process all data in test file | False |

The v2 version adds these arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--num-few-shot` | Number of few-shot examples to use | 4 |
| `--use-hardcoded-examples` | Use hardcoded examples instead of training data | False |
| `--random-seed` | Seed for random example selection | 42 |

## Key Differences Between Versions

### Original Version (`langchain_spam_detector.py`)
- Uses hardcoded few-shot examples only
- Simpler implementation

### Enhanced Version (`langchain_spam_detector_v2.py`)
- Can use examples from training data for few-shot learning
- Randomly selects balanced examples from training data
- Better documentation and error handling
- Outputs statistics about selected examples
- Reproducible with random seed option

## Output

Both detectors will output:
1. Detailed results for each example
2. Overall performance metrics (accuracy, precision, recall, F1)
3. Confusion matrix

Example output:
```
Testing on 20 examples from test data

Evaluating Anthropic Claude model...

Detailed Results:
#   Actual   Predicted  Description
---------------------------------------------------------------------------
1   Not Spam Not Spam   Super cute high waisted blue jeans. recommend...
2   Spam     Spam       Jordan 5 P51 Camo Size 9 Good condition Sole...
...

Anthropic Claude Results:
Accuracy: 0.9000
Precision (Spam): 0.8889
Recall (Spam): 0.8000
F1-Score (Spam): 0.8421

Confusion Matrix:
                 Predicted
                 Not Spam    Spam
Actual Not Spam    10         0
      Spam         2          8
```

## Limitations

- Uses Anthropic API which incurs costs
- By default, processes a small subset of data to save API costs
- Response times depend on API latency

## License

[MIT License](LICENSE)