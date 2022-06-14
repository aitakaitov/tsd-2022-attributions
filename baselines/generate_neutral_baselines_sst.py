import torch
import transformers
from models.bert_512 import BertSequenceClassifierSST
import os
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'

argparser = argparse.ArgumentParser()
argparser.add_argument('--tokenizer')
argparser.add_argument('--model_file')
argparser.add_argument('--output_dir')
args = argparser.parse_args()


tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
model = BertSequenceClassifierSST.from_pretrained(args.tokenizer, num_labels=2)
model.load_state_dict(torch.load(args.model_file))
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

    rnd = torch.randn((1, length, embedding_dimensions), dtype=torch.float32).to(device)
    padding = torch.unsqueeze(padding_embedding.repeat((512 - length, 1)), 0).to(device)
    baseline = torch.cat((rnd, padding), 1).to(device).requires_grad_(True)

    lr = 0.25
    output = model(baseline, attention_mask=attention_mask)[:, 0]
    res = float(output)
    while abs(res - 0.5) > TOLERANCE:
        grads = torch.autograd.grad(output, baseline)[0]
        if res < 0.5:
            baseline = lr * grads + baseline
        else:
            baseline = -1 * lr * grads + baseline

        #lr *= 0.99
        output = model(baseline, attention_mask=attention_mask)[:, 0]
        res = float(output)
        print(output)

    print('Saving')
    torch.save(baseline, OUTPUT_DIR + '/' + f'{length}.pt')


