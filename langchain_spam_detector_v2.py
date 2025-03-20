#!/usr/bin/env python3
"""
LLM Spam Detector (v2)

A spam detector that uses Anthropic's Claude via LangChain to detect keyword spamming
in e-commerce product descriptions.

Key Improvements in v2:
1. Dynamic few-shot examples from training data
2. Configurable number of few-shot examples
3. Option to use either hardcoded or data-driven examples
4. Balanced selection of spam/non-spam examples

Usage examples:
- Basic: python langchain_spam_detector_v2.py
- Use training data: python langchain_spam_detector_v2.py --train-file data/train_set.csv
- More examples: python langchain_spam_detector_v2.py --train-file data/train_set.csv --num-few-shot 8
- All data: python langchain_spam_detector_v2.py --use-all-data
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
import random  # For randomizing example selection
from typing import List, Dict, Tuple

# LangChain and Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load environment variables
load_dotenv()


class AnthropicSpamDetector:
    """
    Spam detector using Anthropic's Claude with LangChain and few-shot learning.
    
    This class uses Anthropic's Claude to detect keyword spamming in product descriptions.
    It supports either hardcoded few-shot examples or dynamic examples from training data.
    
    Attributes:
        llm: LangChain Anthropic client
        system_prompt: System prompt explaining the task to Claude
        few_shot_examples: Examples showing spam vs non-spam classifications
        prompt_template: LangChain prompt template
        chain: LangChain chain for inference
    """
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", few_shot_examples: List[Dict] = None):
        """
        Initialize the spam detector.
        
        Args:
            model: Anthropic model to use
            few_shot_examples: Optional list of examples to use instead of hardcoded ones.
                               Each example should be a dict with 'description' and 'label' keys.
        """
        # Initialize Anthropic client via LangChain
        self.llm = ChatAnthropic(
            model=model,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
        
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
        
        # Default few-shot examples if none provided
        if few_shot_examples is None:
            # Hardcoded few-shot examples as fallback
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
        else:
            # Use the provided examples (typically from training data)
            self.few_shot_examples = few_shot_examples
        
        # Create LangChain prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{user_content}")
        ])
        
        # Create LangChain chain with JSON output parser
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def predict(self, texts: List[str], batch_size: int = 5) -> List[Dict]:
        """
        Predict if texts contain spam using Anthropic's Claude with LangChain.
        
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
        """
        Process a batch of texts with the Anthropic API via LangChain.
        
        This method creates a prompt with few-shot examples for each text,
        sends it to the Claude model, and parses the response.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of classification results
        """
        batch_results = []
        
        # Format few-shot examples for the prompt
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
                
                Respond with a JSON object containing:
                1. "is_spam": boolean indicating if the description contains keyword spam
                2. "spam_score": a number between 0 and 1 indicating the confidence level
                3. "reasoning": brief explanation of your decision
                """
                
                # Invoke the LangChain chain
                response = self.chain.invoke({"user_content": user_content})
                
                # Parse JSON response
                # Find JSON object in the response (Claude sometimes adds extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # Fallback if no JSON found
                    result = {
                        "is_spam": "spam" in response.lower(),
                        "spam_score": 0.5 if "spam" in response.lower() else 0.0,
                        "reasoning": "Failed to parse JSON response"
                    }
                
                batch_results.append(result)
                
            except Exception as e:
                print(f"Error with Anthropic API: {e}")
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


def select_few_shot_examples(texts: List[str], labels: List[int], num_examples: int) -> List[Dict]:
    """
    Select balanced few-shot examples from training data.
    
    This function selects an equal number of spam and non-spam examples
    from the provided texts and labels. It ensures diversity by randomly
    sampling examples from each class.
    
    Args:
        texts: List of product descriptions
        labels: List of corresponding labels (1 for spam, 0 for non-spam)
        num_examples: Total number of examples to select
        
    Returns:
        List of selected examples in dict format
    """
    # Calculate how many examples to select from each class
    examples_per_class = num_examples // 2
    
    # Split data into spam and non-spam
    spam_indices = [i for i, label in enumerate(labels) if label == 1]
    non_spam_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Handle case where we don't have enough examples of one class
    if len(spam_indices) < examples_per_class:
        examples_per_class = len(spam_indices)
        print(f"Warning: Not enough spam examples. Using {examples_per_class} examples per class.")
    
    if len(non_spam_indices) < examples_per_class:
        examples_per_class = len(non_spam_indices)
        print(f"Warning: Not enough non-spam examples. Using {examples_per_class} examples per class.")
    
    # Randomly select examples from each class
    selected_spam_indices = random.sample(spam_indices, examples_per_class)
    selected_non_spam_indices = random.sample(non_spam_indices, examples_per_class)
    
    # Create formatted examples
    few_shot_examples = []
    
    # Add spam examples
    for idx in selected_spam_indices:
        few_shot_examples.append({
            "description": texts[idx],
            "label": "Spam"
        })
    
    # Add non-spam examples
    for idx in selected_non_spam_indices:
        few_shot_examples.append({
            "description": texts[idx],
            "label": "Not Spam"
        })
    
    # Shuffle examples
    random.shuffle(few_shot_examples)
    
    return few_shot_examples


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
    
    print("\nAnthropic Claude Results:")
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
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='LLM Spam Detector using Anthropic Claude (v2)')
    parser.add_argument('--test-file', type=str, default='data/test_set.csv',
                      help='Path to test data CSV file (default: data/test_set.csv)')
    parser.add_argument('--train-file', type=str, default=None,
                      help='Path to training data CSV file (default: not used)')
    parser.add_argument('--num-examples', type=int, default=20,
                      help='Number of examples to process (default: 20)')
    parser.add_argument('--use-all-data', action='store_true',
                      help='Use all data in the test file instead of a subset')
    parser.add_argument('--num-few-shot', type=int, default=4,
                      help='Number of few-shot examples to use (default: 4)')
    parser.add_argument('--use-hardcoded-examples', action='store_true',
                      help='Use hardcoded few-shot examples instead of examples from training data')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for example selection (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    print("Loading data...")
    # Load test data
    test_texts, test_labels = load_data(args.test_file)
    
    # Determine test subset size
    test_subset_size = len(test_texts) if args.use_all_data else min(args.num_examples, len(test_texts))
    test_subset_texts = test_texts[:test_subset_size]
    test_subset_labels = test_labels[:test_subset_size]
    
    print(f"Testing on {test_subset_size} examples from test data")
    
    # Set up few-shot examples
    few_shot_examples = None  # Default to hardcoded examples
    
    # Load and use training data if specified
    if args.train_file and not args.use_hardcoded_examples:
        print(f"Loading training data from {args.train_file}")
        train_texts, train_labels = load_data(args.train_file)
        
        # Use the improved selection function for balanced, randomized examples
        few_shot_examples = select_few_shot_examples(
            train_texts, 
            train_labels, 
            args.num_few_shot
        )
        
        print(f"Using {len(few_shot_examples)} few-shot examples from training data")
        print(f"- Spam examples: {sum(1 for ex in few_shot_examples if ex['label'] == 'Spam')}")
        print(f"- Non-spam examples: {sum(1 for ex in few_shot_examples if ex['label'] == 'Not Spam')}")
    else:
        if args.train_file:
            print("Using hardcoded few-shot examples (--use-hardcoded-examples flag is set)")
        else:
            print("Using hardcoded few-shot examples (no training file specified)")
    
    # Anthropic model
    print("\nEvaluating Anthropic Claude model...")
    anthropic_detector = AnthropicSpamDetector(few_shot_examples=few_shot_examples)
    anthropic_predictions = anthropic_detector.predict_binary(test_subset_texts)
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"{'#':<3} {'Actual':<8} {'Predicted':<10} {'Description':<50}")
    print("-" * 75)
    
    for i, (text, true_label, pred_label) in enumerate(zip(test_subset_texts, test_subset_labels, anthropic_predictions)):
        # Truncate text for display
        short_text = text[:47] + "..." if len(text) > 50 else text
        print(f"{i+1:<3} {'Spam' if true_label == 1 else 'Not Spam':<8} {'Spam' if pred_label == 1 else 'Not Spam':<10} {short_text:<50}")
    
    # Evaluate model
    metrics = evaluate_model(test_subset_labels, anthropic_predictions)
    
    if not args.use_all_data:
        print("\nNote: Anthropic Claude model was evaluated on a subset of the test data to save API costs.")
        print("Use --use-all-data to process the entire dataset.")


if __name__ == "__main__":
    main() 