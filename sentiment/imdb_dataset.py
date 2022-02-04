import torch
from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, max_length=512):
        self.dataset_size = 0
        self.loaded_dataset = []
        self._max_length = max_length

        with open('sentiment_train/imdb50k/dataset_cleaned.json', 'r', encoding='utf-8') as dataset_f:
            self.loaded_dataset = json.load(dataset_f)
            self.dataset_size = len(self.loaded_dataset)

    def __len__(self):
        return 500#self.dataset_size

    def str_to_label(self, _str):
        if _str == 'positive':
            return 1
        else:
            return 0

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return torch.tensor(self.str_to_label(self.loaded_dataset[idx][1]), dtype=torch.float32)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        encoded = tokenizer(self.loaded_dataset[idx][0], padding='max_length', max_length=self._max_length,
                            truncation=True, return_tensors='pt')
        return encoded.data['input_ids'], encoded.data['attention_mask']

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
