import json

'''
Create a phrase -> sentiment dictionary
'''

dictionary_file = 'dictionary.txt'
sentiment_file = 'sentiment_labels.txt'
phrase_sentiment_file = 'phrase_sentiments.json'

id_phrase_dict = {}
id_sentiment_dict = {}
phrase_sentiment_dict = {}

with open(dictionary_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        phrase, _id = line.split('|')
        _id = int(_id[0:len(_id) - 1])
        id_phrase_dict[_id] = phrase

with open(sentiment_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[1:]:
        _id, sentiment = line.split('|')
        sentiment = sentiment[0:len(sentiment) - 1]
        id_sentiment_dict[int(_id)] = float(sentiment)

for _id, phrase in id_phrase_dict.items():
    phrase_sentiment_dict[phrase] = id_sentiment_dict[_id]

with open(phrase_sentiment_file, 'w+', encoding='utf-8') as f:
    f.write(json.dumps(phrase_sentiment_dict))
