import os

import torch
import transformers
from models.bert_512 import BertSequenceClassifierSST
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--tokenizer')
argparser.add_argument('--model_folder')
argparser.add_argument('--output_dir')
args = argparser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
model = BertSequenceClassifierSST.from_pretrained(args.model_folder, num_labels=2, local_files_only=True)
model = model.to(device)
model.eval()
embeddings = model.bert.base_model.embeddings.word_embeddings.weight.data
embedding_dimensions = embeddings.shape[1]
padding_embedding = tokenizer.convert_tokens_to_ids('[PAD]')
padding_embedding = torch.index_select(embeddings, 0, torch.tensor(padding_embedding).to(device))

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR)

TOLERANCE = 0.025

for length in range(1, 513):
    print('Length ' + str(length))
    arr = [1 for i in range(length)]
    arr.extend([0 for i in range(512 - length)])
    attention_mask = torch.tensor([arr]).to(device)

    rnd = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to('cpu')
    padding = torch.unsqueeze(padding_embedding.repeat((512 - length, 1)), 0).to('cpu')
    baseline = torch.cat((rnd, padding), 1).to(device).requires_grad_(True)

    output = model(baseline, attention_mask=attention_mask)[:, 0]
    res = float(output)
    while abs(res - 0.5) > TOLERANCE:
        rnd = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to('cpu')
        padding = torch.unsqueeze(padding_embedding.repeat((512 - length, 1)), 0).to('cpu')
        baseline = torch.cat((rnd, padding), 1).to(device).requires_grad_(True)

        output = model(baseline, attention_mask=attention_mask)[:, 0]
        res = float(output)

    print('Saving')
    torch.save(baseline, OUTPUT_DIR + '/' + f'{length}.pt')


