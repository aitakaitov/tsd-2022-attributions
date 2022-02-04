from transformers import AutoTokenizer

import json


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

with open('dataset_cleaned.json', 'r', encoding='utf-8') as f:
    text_label_list = json.loads(f.read())

average_count = 0
max_count = -1
max_text = ""

for text, label in text_label_list[1:]:
    tokenized = tokenizer(text)
    average_count += len(tokenized.data['input_ids'])
    if max_count < len(tokenized.data['input_ids']):
        max_count = len(tokenized.data['input_ids'])
        max_text = text

average_count /= 50000

print("average count: " + str(average_count))
print("max count: " + str(max_count))
print(max_text)