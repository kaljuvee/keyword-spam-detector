#!/usr/bin/env python3
"""
Convert Jupyter Notebook HTML to Python Script

This script demonstrates how to use the html_to_python_converter module
to convert an HTML file exported from a Jupyter notebook to a Python script.
"""

import os
from tests.html_to_python_converter import extract_code_from_html, save_as_python_script

def main():
    # Path to the HTML file
    html_file = "Classifying Keyword Spamming-v2.html"
    
    # Output Python file
    output_file = "classifying_keyword_spamming.py"
    
    # Check if the HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: File {html_file} not found")
        return
    
    # Extract code cells from the HTML file
    print(f"Extracting code from {html_file}...")
    code_cells = extract_code_from_html(html_file)
    
    if not code_cells:
        print("Warning: No code cells found in the HTML file")
        return
    
    # Save the code cells as a Python script
    save_as_python_script(code_cells, output_file)
    print(f"Successfully extracted {len(code_cells)} code cells to {output_file}")

if __name__ == "__main__":
    main() 