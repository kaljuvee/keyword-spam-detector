"""
Keyword Spam Detector

This module provides functionality to detect keyword spamming in product descriptions
using both classical ML (Random Forest) and LLM-based approaches.
"""

import os
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# OpenAI
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))


class TextPreprocessor:
    """Class for text preprocessing operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing special characters, extra spaces, and converting to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [word for word in tokens if word.lower() not in stop_words]
    
    @staticmethod
    def lemmatize(text: str) -> str:
        """
        Lemmatize text using spaCy.
        
        Args:
            text: Input text string
            
        Returns:
            Lemmatized text string
        """
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    
    @classmethod
    def preprocess(cls, text: str, remove_stop: bool = True, lemmatize: bool = True) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text string
            remove_stop: Whether to remove stopwords
            lemmatize: Whether to lemmatize text
            
        Returns:
            Preprocessed text string
        """
        text = cls.clean_text(text)
        
        if lemmatize:
            text = cls.lemmatize(text)
        
        if remove_stop:
            tokens = cls.tokenize(text)
            tokens = cls.remove_stopwords(tokens)
            text = " ".join(tokens)
        
        return text


class SpamFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features for spam detection."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            # Preprocess text
            clean_text = self.preprocessor.clean_text(text)
            
            # Extract features
            features.append({
                'text_length': len(text),
                'clean_text_length': len(clean_text),
                'word_count': len(clean_text.split()),
                'hashtag_count': text.count('#'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
                'hashtag_to_text_ratio': text.count('#') / max(len(text.split()), 1),
                'contains_ignore': 1 if 'ignore' in text.lower() else 0,
                'contains_hashtags_at_end': 1 if re.search(r'#\w+\s*$', text) else 0,
                'hashtag_section_length': len(text.split('#')[-1]) if '#' in text else 0,
                'brand_count': sum(1 for brand in ['nike', 'adidas', 'gucci', 'chanel', 'versace', 'supreme', 'zara', 'h&m', 'brandy', 'melville'] if brand in text.lower()),
                'tag_indicators': sum(1 for tag in ['tag', 'tags:', 'tags', 'tagged', 'ignore:', 'ignore'] if tag in text.lower()),
                'keyword_density': len(set(clean_text.split())) / max(len(clean_text.split()), 1),
                'has_tag_section': 1 if any(marker in text.lower() for marker in ['tags:', 'ignore:', '~ignore~', 'tagged for exposure']) else 0
            })
        
        return pd.DataFrame(features)


class RandomForestSpamDetector:
    """Spam detector using Random Forest classifier."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.feature_extractor = SpamFeatureExtractor()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model_trained = False
    
    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from text data.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame of extracted features
        """
        # TF-IDF features
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        tfidf_features = self.tfidf_vectorizer.transform(preprocessed_texts)
        
        # Custom features
        custom_features = self.feature_extractor.transform(texts)
        
        # Combine features
        tfidf_df = pd.DataFrame(tfidf_features.toarray())
        
        return pd.concat([tfidf_df, custom_features], axis=1)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        """
        Train the spam detector.
        
        Args:
            texts: List of text strings
            labels: List of binary labels (0 for non-spam, 1 for spam)
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(preprocessed_texts)
        
        # Extract features
        X = self.extract_features(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.model_trained = True
        
        print(f"Random Forest model trained on {len(texts)} examples")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict spam probability for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of spam probabilities
        """
        if not self.model_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Extract features
        X = self.extract_features(texts)
        
        # Predict probabilities
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict_binary(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions (spam/not spam).
        
        Args:
            texts: List of text strings
            threshold: Probability threshold for classification
            
        Returns:
            Array of binary predictions (0 for non-spam, 1 for spam)
        """
        probs = self.predict(texts)
        return (probs >= threshold).astype(int)


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
                    "reasoning": "Error processing with LLM"
                })
        
        return batch_results
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict spam probability for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of spam probabilities
        """
        results = self.predict(texts)
        return np.array([result.get("spam_score", 0.0) for result in results])
    
    def predict_binary(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions (spam/not spam).
        
        Args:
            texts: List of text strings
            threshold: Probability threshold for classification
            
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


def evaluate_model(y_true: List[int], y_pred: List[int], model_name: str) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Spam): {report['1']['precision']:.4f}")
    print(f"Recall (Spam): {report['1']['recall']:.4f}")
    print(f"F1-Score (Spam): {report['1']['f1-score']:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Not Spam    Spam")
    print(f"Actual Not Spam    {cm[0][0]}         {cm[0][1]}")
    print(f"      Spam         {cm[1][0]}         {cm[1][1]}")
    
    return {
        'accuracy': accuracy,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main function to run the spam detection comparison."""
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
    
    # OpenAI model
    print("\nEvaluating OpenAI model...")
    openai_detector = OpenAISpamDetector()
    openai_predictions = openai_detector.predict_binary(test_texts)
    openai_metrics = evaluate_model(test_labels, openai_predictions, "OpenAI LLM")
    
    # Compare results
    print("\nModel Comparison:")
    print(f"{'Metric':<15} {'Random Forest':<15} {'OpenAI LLM':<15}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15} {rf_metrics['accuracy']:<15.4f} {openai_metrics['accuracy']:<15.4f}")
    print(f"{'Precision':<15} {rf_metrics['precision']:<15.4f} {openai_metrics['precision']:<15.4f}")
    print(f"{'Recall':<15} {rf_metrics['recall']:<15.4f} {openai_metrics['recall']:<15.4f}")
    print(f"{'F1-Score':<15} {rf_metrics['f1_score']:<15.4f} {openai_metrics['f1_score']:<15.4f}")


if __name__ == "__main__":
    main() 