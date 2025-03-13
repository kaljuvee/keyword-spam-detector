#!/usr/bin/env python3
"""
Run the keyword spam detector on the provided datasets and compare results.
"""

from spam_detector import (
    RandomForestSpamDetector,
    OpenAISpamDetector,
    load_data,
    evaluate_model
)

def main():
    """Run the spam detection comparison."""
    print("Loading data...")
    train_texts, train_labels = load_data('data/train_set.csv')
    test_texts, test_labels = load_data('data/test_set.csv')
    
    print(f"Train data: {len(train_texts)} examples")
    print(f"Test data: {len(test_texts)} examples")
    
    # Random Forest model
    print("\nTraining Random Forest model...")
    rf_detector = RandomForestSpamDetector()
    rf_detector.train(train_texts, train_labels)
    
    print("Evaluating Random Forest model...")
    rf_predictions = rf_detector.predict_binary(test_texts)
    rf_metrics = evaluate_model(test_labels, rf_predictions, "Random Forest")
    
    # OpenAI model (using a smaller subset for testing to save API costs)
    print("\nEvaluating OpenAI model...")
    # Use a smaller subset for OpenAI to save API costs
    test_subset_size = min(50, len(test_texts))
    test_subset_texts = test_texts[:test_subset_size]
    test_subset_labels = test_labels[:test_subset_size]
    
    openai_detector = OpenAISpamDetector()
    openai_predictions = openai_detector.predict_binary(test_subset_texts)
    openai_metrics = evaluate_model(test_subset_labels, openai_predictions, "OpenAI LLM")
    
    # Compare results
    print("\nModel Comparison:")
    print(f"{'Metric':<15} {'Random Forest':<15} {'OpenAI LLM':<15}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15} {rf_metrics['accuracy']:<15.4f} {openai_metrics['accuracy']:<15.4f}")
    print(f"{'Precision':<15} {rf_metrics['precision']:<15.4f} {openai_metrics['precision']:<15.4f}")
    print(f"{'Recall':<15} {rf_metrics['recall']:<15.4f} {openai_metrics['recall']:<15.4f}")
    print(f"{'F1-Score':<15} {rf_metrics['f1_score']:<15.4f} {openai_metrics['f1_score']:<15.4f}")
    
    print("\nNote: OpenAI model was evaluated on a subset of the test data to save API costs.")


if __name__ == "__main__":
    main() 