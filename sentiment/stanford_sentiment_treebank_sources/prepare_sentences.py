import json

'''
Create a phrase -> sentiment dictionary
'''

sentences_file = 'SOStr.txt'
cleaned_file = 'sentences_tokens.json'

with open(sentences_file, 'r', encoding='utf-8') as f:
    sentence_lines = f.readlines()

sentences_list = []
for line in sentence_lines:
    split_text = line.split('|')
    text = ""
    for i in range(len(split_text)):
        if i == 0:
            text += split_text[i]
        else:
            text += ' ' + split_text[i]

        if i == len(split_text) - 1:
            split_text[i] = split_text[i][:len(split_text[i]) - 1]

    sentences_list.append([text[0:len(text) - 1], split_text])

with open(cleaned_file, 'w+', encoding='utf-8') as f:
    f.write(json.dumps(sentences_list))
