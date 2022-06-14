import random

import torch
import numpy as np
from torch.nn.functional import one_hot
from sklearn.model_selection import StratifiedKFold


class NewsDataset(torch.utils.data.Dataset):

    def __init__(self, data: str, tokenizer, classes_dict):
        self.labels = []
        self.texts = []
        self.tokenizer = tokenizer
        self.class_dict = classes_dict
        self.baseline_sample_count = 500
        self.folds = []
        self.mode = 'train'
        self.other_labels = []
        self.other_texts = []
        self.first_train_fold_index = 0

        for line in data.split('\n'):
            split_line = line.split('~')

            text = split_line[0]
            encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            self.texts.append(
                [encoded.data['input_ids'], encoded.data['attention_mask']]
            )

            classes = split_line[1:]
            classes = [self.class_dict[clss] for clss in classes]
            label = torch.sum(one_hot(torch.tensor(classes, dtype=torch.long), len(self.class_dict.items())), dim=0)
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def _add_baseline_samples(self):
        encoded = self.tokenizer("", padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        input_ids_list = encoded.data['input_ids'][0].numpy().tolist()
        pad_token_index = input_ids_list[2]
        cls_token_index = input_ids_list[0]
        sep_token_index = input_ids_list[1]
        baseline_input_ids = torch.tensor([pad_token_index for _ in range(512)], dtype=torch.int)
        baseline_input_ids[0] = cls_token_index
        baseline_input_ids[511] = sep_token_index
        attention_mask = torch.tensor([1 for _ in range(512)], dtype=torch.int)
        baseline_label_index = self.class_dict['baseline']
        baseline_label = [0 for _ in self.class_dict.items()]
        baseline_label[baseline_label_index] = 1
        baseline_label = torch.tensor(baseline_label, dtype=torch.int)
        for i in range(self.baseline_sample_count):
            self.texts.append([baseline_input_ids, attention_mask])
            self.labels.append(baseline_label)

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

