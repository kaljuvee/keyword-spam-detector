#!/usr/bin/env python
"""
Setup script for keyword spam detector project.
This script installs all required packages and downloads necessary models.
"""

import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def download_nltk_data():
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')

def main():
    install_requirements()
    download_spacy_model()
    download_nltk_data()
    print("\nSetup completed successfully!")
    print("You can now run the spam detector with: python classifying_keyword_spamming.py")

if __name__ == "__main__":
    main() 