#!/usr/bin/env python3
"""
HTML to Python Converter

This script converts an HTML file exported from a Jupyter notebook to a Python script.
It extracts all code cells from the HTML and saves them as a Python script.
"""

import sys
import re
import os
from bs4 import BeautifulSoup
import argparse

def extract_code_from_html(html_file):
    """
    Extract Python code from HTML file exported from Jupyter notebook
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all code cells
    # Different Jupyter exports might use different class names
    code_cells = []
    
    # Try different patterns for code cells
    for div in soup.find_all('div', class_=lambda c: c and ('jp-CodeCell' in c or 'code_cell' in c)):
        code_element = div.find('div', class_=lambda c: c and ('jp-Cell-inputArea' in c or 'input_area' in c))
        if code_element:
            code = code_element.find('pre')
            if code:
                code_cells.append(code.text)
    
    # If the above didn't work, try another approach with pre tags
    if not code_cells:
        for pre in soup.find_all('pre'):
            # Check if this looks like Python code
            if pre.text.strip() and (
                re.search(r'import\s+\w+', pre.text) or 
                re.search(r'def\s+\w+\s*\(', pre.text) or
                re.search(r'class\s+\w+\s*\(', pre.text) or
                re.search(r'^\s*#', pre.text, re.MULTILINE)
            ):
                code_cells.append(pre.text)
    
    # If still no code found, try one more approach
    if not code_cells:
        # Look for code in div with class containing 'highlight'
        for div in soup.find_all('div', class_=lambda c: c and 'highlight' in c):
            code = div.find('pre')
            if code:
                code_cells.append(code.text)
    
    return code_cells

def save_as_python_script(code_cells, output_file):
    """
    Save extracted code cells as a Python script
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\nThis script was automatically generated from a Jupyter notebook HTML export\n"""\n\n')
        
        for i, cell in enumerate(code_cells):
            if i > 0:
                f.write('\n\n')
            f.write(cell)

def main():
    parser = argparse.ArgumentParser(description='Convert HTML exported Jupyter notebook to Python script')
    parser.add_argument('html_file', help='Path to the HTML file')
    parser.add_argument('-o', '--output', help='Output Python file (default: same name with .py extension)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.html_file):
        print(f"Error: File {args.html_file} not found")
        sys.exit(1)
    
    if not args.output:
        args.output = os.path.splitext(args.html_file)[0] + '.py'
    
    code_cells = extract_code_from_html(args.html_file)
    
    if not code_cells:
        print("Warning: No code cells found in the HTML file")
        sys.exit(1)
    
    save_as_python_script(code_cells, args.output)
    print(f"Successfully extracted {len(code_cells)} code cells to {args.output}")

if __name__ == '__main__':
    main() 