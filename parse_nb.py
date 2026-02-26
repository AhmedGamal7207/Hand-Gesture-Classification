import json
import re

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        lines = src.split('\n')
        for line in lines:
            line = line.strip()
            # We want to match headers like:
            # "1. Loading and Exploring Data"
            # "1.1 Importing Libraries"
            # "# 4. Model Training"
            if re.match(r'^#*\s*\d+(\.\d+)?\s+[A-Za-z]', line):
                print(re.sub(r'<[^>]+>', '', line))
