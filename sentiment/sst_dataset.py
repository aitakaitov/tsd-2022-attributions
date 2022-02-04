import torch
import numpy as np
from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {
    'positive': 1,
    'negative': 0
}


class SSTDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.labels = []
        self.texts = []

        with open('stanford_sentiment_treebank_sources/phrase_sentiments.json', 'r', encoding='utf-8') as sentiments_f:
            text_sentiment_dict = json.load(sentiments_f)

        with open('stanford_sentiment_treebank_sources/sentences_tokens.json', 'r', encoding='utf-8') as sentences_f:
            data = json.load(sentences_f)

        for sentence, tokens in data:
            encoded = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            self.texts.append(
                [encoded.data['input_ids'], encoded.data['attention_mask']]
            )
            self.labels.append(torch.tensor(text_sentiment_dict[sentence], dtype=torch.float32))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y