#!/usr/bin/env python3
"""
LLM Spam Detector

A simplified spam detector that uses only OpenAI's LLM to detect keyword spamming
in e-commerce product descriptions.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# OpenAI
import openai
from dotenv import load_dotenv

# Scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAISpamDetector:
    """Spam detector using OpenAI's LLM with few-shot learning."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.system_prompt = """
        You are an expert at detecting keyword spam in e-commerce product descriptions.
        
        Keyword spamming is when sellers include unrelated, irrelevant keywords in their product descriptions
        to improve their ranking in search results. This creates a poor experience for buyers who see irrelevant
        products in their search results.
        
        Examples of keyword spam:
        1. Adding a list of brand names that aren't related to the actual product
        2. Adding hashtags with irrelevant terms
        3. Including phrases like "ignore:" followed by keywords
        4. Adding a long list of unrelated terms at the end of a description
        
        Your task is to analyze product descriptions and determine if they contain keyword spam.
        Respond with a JSON object containing:
        1. "is_spam": boolean indicating if the description contains keyword spam
        2. "spam_score": a number between 0 and 1 indicating the confidence level
        3. "reasoning": brief explanation of your decision
        """
        
        # Few-shot examples for better performance
        self.few_shot_examples = [
            {
                "description": "Super cute high waisted blue jeans. recommended for shorter / petite girls, I'm 5'1 and it fits me perfect. These are more of jeggings than jeans and have a polyester material inside. Only flaw: the zipper goes down by itself sometimes. Size 0 / waist 23",
                "label": "Not Spam"
            },
            {
                "description": "Jordan 5 P51 Camo Size 9 Good condition Soles have yellowing No box or lacelocks $160 Travis Supreme concepts Jordan 1 3 5 6 7 11 12 Nike ovo Kanye yeezy boost 350 shadow royal bred shattered cement top 3 black toe infrared raptor gamma space jam air max vapormax flyknit Travis Scott kaws off white atmos 95 97 98 silver gold bullet protro Kobe fieg kith Levi's undefeated Palace tinker stash SB dunk stone island Foamposite plus Acronym VF Wotherspoon 270 SW LeBron Kyrie Pippen",
                "label": "Spam"
            },
            {
                "description": "Denim jacket Never been worn Size XL",
                "label": "Not Spam"
            },
            {
                "description": "gorgeous y2k juicy couture velvet zip up. super pretty turquoise color. worn only a few times and in perfect condition. 100% authentic! size medium but would fit a small nicely tags #lingerie #lace #vintagedepop #sofuckingdepop #depopfamous lily rose depp devon lee carlson corset black tank cami 90s 2000s y2k tank top camisole black lace top mock lingerie black v neck lace lined top strappy top carrie bradshaw milkmaid cami crop top blouse v neck lace sex and the city cruel intentions clueless black swan victorias secret cami",
                "label": "Spam"
            }
        ]
    
    def predict(self, texts: List[str], batch_size: int = 5) -> List[Dict]:
        """
        Predict if texts contain spam using OpenAI's LLM with few-shot learning.
        
        Args:
            texts: List of text strings
            batch_size: Number of examples to process in each API call
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        # Process in batches to avoid token limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self._predict_batch(batch_texts)
            results.extend(batch_results)
            
        return results
    
    def _predict_batch(self, texts: List[str]) -> List[Dict]:
        """Process a batch of texts with the OpenAI API."""
        batch_results = []
        
        # Create few-shot examples content
        few_shot_content = "\n\n".join([
            f"Description: {example['description']}\nLabel: {example['label']}"
            for example in self.few_shot_examples
        ])
        
        for text in texts:
            try:
                # Create prompt with few-shot examples
                user_content = f"""
                Here are some examples of spam and non-spam product descriptions:
                
                {few_shot_content}
                
                Now, analyze this product description for keyword spam:
                
                Description: {text}
                """
                
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content)
                batch_results.append(result)
                
            except Exception as e:
                print(f"Error with OpenAI API: {e}")
                # Fallback result
                batch_results.append({
                    "is_spam": False,
                    "spam_score": 0.0,
                    "reasoning": f"Error processing with LLM: {str(e)}"
                })
        
        return batch_results
    
    def predict_binary(self, texts: List[str]) -> np.ndarray:
        """
        Make binary predictions (spam/not spam).
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of binary predictions (0 for non-spam, 1 for spam)
        """
        results = self.predict(texts)
        return np.array([1 if result.get("is_spam", False) else 0 for result in results])


def load_data(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (descriptions, labels)
    """
    df = pd.read_csv(file_path)
    return df['description'].tolist(), df['label'].tolist()


def evaluate_model(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Evaluate model performance using scikit-learn metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate metrics using scikit-learn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix using scikit-learn
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nOpenAI LLM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Spam): {precision:.4f}")
    print(f"Recall (Spam): {recall:.4f}")
    print(f"F1-Score (Spam): {f1:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Not Spam    Spam")
    print(f"Actual Not Spam    {cm[0][0]}         {cm[0][1]}")
    print(f"      Spam         {cm[1][0]}         {cm[1][1]}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main function to run the LLM spam detection."""
    print("Loading data...")
    # Load a small subset of test data to save API costs
    test_texts, test_labels = load_data('data/test_set.csv')
    
    # Use a smaller subset for testing
    test_subset_size = min(20, len(test_texts))
    test_subset_texts = test_texts[:test_subset_size]
    test_subset_labels = test_labels[:test_subset_size]
    
    print(f"Testing on {test_subset_size} examples from test data")
    
    # OpenAI model
    print("\nEvaluating OpenAI model...")
    openai_detector = OpenAISpamDetector()
    openai_predictions = openai_detector.predict_binary(test_subset_texts)
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"{'#':<3} {'Actual':<8} {'Predicted':<10} {'Description':<50}")
    print("-" * 75)
    
    for i, (text, true_label, pred_label) in enumerate(zip(test_subset_texts, test_subset_labels, openai_predictions)):
        # Truncate text for display
        short_text = text[:47] + "..." if len(text) > 50 else text
        print(f"{i+1:<3} {'Spam' if true_label == 1 else 'Not Spam':<8} {'Spam' if pred_label == 1 else 'Not Spam':<10} {short_text:<50}")
    
    # Evaluate model
    metrics = evaluate_model(test_subset_labels, openai_predictions)
    
    print("\nNote: OpenAI model was evaluated on a small subset of the test data to save API costs.")


if __name__ == "__main__":
    main() 