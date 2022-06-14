import argparse
import os

import transformers
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import tqdm
from datasets_ours.news.news_dataset import NewsDataset
import json

from sys import argv


def get_class_dict():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())
    return class_dict


def get_file_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def train(learning_rate, epochs, model):
    train = NewsDataset(get_file_text('datasets_ours/news/train.csv'), tokenizer, classes_dict)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # optimization
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    sigmoid = torch.nn.Sigmoid()

    # metrics
    train_metric = torchmetrics.F1Score().to(device)
    writer = SummaryWriter(output_dir + '/logs')

    # cuda
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # training, eval
    for epoch_num in range(epochs):
        print(f'EPOCH: {epoch_num + 1}')
        iteration = 0
        model.train()
        scheduler.step()
        for train_input, train_label in tqdm.tqdm(train_dataloader):
            train_label = torch.unsqueeze(train_label, dim=-1)
            train_label = train_label.to(device)
            mask = torch.squeeze(train_input[1].to(device), dim=0)
            input_id = train_input[0].squeeze(1).to(device)

            output = model(input_id, mask).logits

            with torch.autocast('cuda'):
                batch_loss = criterion(output, train_label)

            writer.add_scalar('loss/train', float(batch_loss), epoch_num * len(train) + iteration)
            train_metric(sigmoid(output), torch.tensor(train_label, dtype=torch.int32))

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            iteration += 1

        writer.add_scalar('accuracy/train', train_metric.compute(), (epoch_num + 1) * len(train))
        print(f'F1 TRAIN: {float(train_metric.compute())}')
        train_metric.reset()
        model.eval()
    torch.save(model, output_dir + f'/saved-model')


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=4, help="Number of training epochs")
parser.add_argument("--lr", default=1e-5, help="Learning rate")
parser.add_argument("--model_name", default='UWB-AIR/Czert-B-base-cased', help="Pretrained model path")
parser.add_argument("--batch_size", default=4, help="Batch size")
parser.add_argument("--output_dir", default='kfold-training-output', help="Output directory")

args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
model_name = args.model_name
batch_size = args.batch_size
output_dir = args.output_dir


try:
    os.mkdir(output_dir)
except OSError:
    pass

classes_dict = get_class_dict()
model = transformers.BertForSequenceClassification.from_pretrained(model_name, num_labels=len(classes_dict), local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

train(LR, EPOCHS, model)
