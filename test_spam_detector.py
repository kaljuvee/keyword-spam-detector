#!/usr/bin/env python
"""
Test script for keyword spam detector.

This script demonstrates the functionality of the keyword spam detector
using both classical ML and LLM-based approaches.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from spam_detector import (
    TextPreprocessor,
    ClassicalSpamDetector,
    LLMSpamDetector,
    HybridSpamDetector
)

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. LLM-based detection will not work.")
    print("Please create a .env file with your OpenAI API key.")
    USE_LLM = False
else:
    USE_LLM = True


def load_example_data() -> List[Dict]:
    """
    Load example data for testing.
    
    Returns:
        List of dictionaries with text and label
    """
    # Example spam descriptions from the problem statement
    spam_examples = [
        {
            "text": """Low waist/rise diesel bootcut/flared jeans. Size XS/6. Great condition. Cool red stitching details.
Message for any questions :)Uk shipping only
No returns
#vintage #diesel #denim #lowrise #levi #wrangler #lee #y2k #90s #2010s #blue #black #fade""",
            "label": 1  # Spam
        },
        {
            "text": """Low rise y2k blue Diesel bootcut jeans
Size label W29 L32
Flat laid measurements below —
32 inch waist ( sits on hips )
7 inch rise
32 inch inseamFREE UK SHIP
£15 international
Ignore: 80s 90s y2k baggy navy jeans denim levi calvin klein""",
            "label": 1  # Spam
        }
    ]
    
    # Non-spam examples (created for testing)
    non_spam_examples = [
        {
            "text": """Diesel bootcut jeans in excellent condition. Size XS/6 with low waist design. 
Features cool red stitching details that make these jeans stand out.
Message me with any questions. UK shipping only. No returns accepted.""",
            "label": 0  # Not spam
        },
        {
            "text": """Authentic Diesel bootcut jeans in blue. Low rise design perfect for a y2k look.
Size W29 L32. Measurements: 32 inch waist (sits on hips), 7 inch rise, 32 inch inseam.
FREE shipping within the UK. International shipping available for £15.""",
            "label": 0  # Not spam
        },
        {
            "text": """Brand new Levi's 501 Original Fit Men's Jeans. Classic straight leg design with button fly.
Size 32W x 34L. Dark blue wash, 100% cotton denim. 
These iconic jeans offer timeless style and durability.
Free shipping on orders over £50.""",
            "label": 0  # Not spam
        }
    ]
    
    # Combine examples
    return spam_examples + non_spam_examples


def evaluate_detector(detector, texts: List[str], labels: List[int], name: str) -> Dict:
    """
    Evaluate a detector on test data.
    
    Args:
        detector: Spam detector instance
        texts: List of text strings
        labels: List of binary labels (0 for non-spam, 1 for spam)
        name: Name of the detector for reporting
        
    Returns:
        Dictionary with evaluation results
    """
    # Make predictions
    predictions = detector.predict_binary(texts)
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    
    # Calculate precision, recall, and F1 score for spam class
    true_positives = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    false_positives = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    false_negatives = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """Main function to run the test script."""
    print("Keyword Spam Detector Test")
    print("==========================")
    
    # Load example data
    examples = load_example_data()
    texts = [example["text"] for example in examples]
    labels = [example["label"] for example in examples]
    
    # Split data for training and testing
    # In a real scenario, we would have more data and use a proper train/test split
    train_texts = texts
    train_labels = labels
    test_texts = texts
    test_labels = labels
    
    # Initialize detectors
    classical_detector = ClassicalSpamDetector()
    
    # Train classical detector
    print("\nTraining classical detector...")
    classical_detector.train(train_texts, train_labels)
    
    # Evaluate classical detector
    classical_results = evaluate_detector(
        classical_detector, test_texts, test_labels, "Classical Detector"
    )
    
    # Initialize and evaluate LLM detector if API key is available
    if USE_LLM:
        print("\nTesting LLM detector...")
        llm_detector = LLMSpamDetector()
        llm_results = evaluate_detector(
            llm_detector, test_texts, test_labels, "LLM Detector"
        )
        
        # Initialize and evaluate hybrid detector
        print("\nTesting hybrid detector...")
        hybrid_detector = HybridSpamDetector(
            classical_weight=0.4,
            llm_weight=0.6,
            classical_detector=classical_detector,
            llm_detector=llm_detector
        )
        hybrid_results = evaluate_detector(
            hybrid_detector, test_texts, test_labels, "Hybrid Detector"
        )
        
        # Get detailed predictions from hybrid detector
        detailed_results = hybrid_detector.predict_with_explanation(test_texts)
        
        # Print detailed results
        print("\nDetailed Results:")
        for i, result in enumerate(detailed_results):
            print(f"\nExample {i+1}:")
            print(f"Text: {result['text'][:100]}...")
            print(f"True Label: {'Spam' if test_labels[i] == 1 else 'Not Spam'}")
            print(f"Prediction: {'Spam' if result['is_spam'] else 'Not Spam'}")
            print(f"Combined Score: {result['combined_score']:.2f}")
            print(f"Classical Score: {result['classical_score']:.2f}")
            print(f"LLM Score: {result['llm_score']:.2f}")
            print(f"LLM Reasoning: {result['llm_reasoning']}")
    else:
        print("\nSkipping LLM and hybrid detector tests (no API key)")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main() 