import json
import re

import pandas as pd


def clean_text(text: str):
    text = re.sub('(<br />)+', '', text)
    text = re.sub('""', '"', text)
    return text


df = pd.read_csv('dataset_raw.csv', sep=',')
new_csv_lines = []

for index, row in df.iterrows():
    new_csv_lines.append([clean_text(row['review']), row['sentiment']])

with open('dataset_cleaned.json', 'w+', encoding='utf-8') as f:
    f.write(json.dumps(new_csv_lines))