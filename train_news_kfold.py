import os

import transformers
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import tqdm
from datasets_ours.news.news_dataset import NewsDataset
import json
from sklearn.model_selection import StratifiedKFold, KFold

import argparse


def get_class_dict():
    with open('datasets_ours/news/classes.json', 'r', encoding='utf-8') as f:
        class_dict = json.loads(f.read())
    return class_dict


def get_file_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_fold_sizes(dataset_length, fold_count=5):
    fold_sizes = []
    remaining = dataset_length
    for k in range(fold_count):
        if k < fold_count - 1:
            fold_sizes.append(int(dataset_length / fold_count))
            remaining -= int(dataset_length / fold_count)
        else:
            fold_sizes.append(remaining)
    return fold_sizes


def train(learning_rate, epochs):
    train = NewsDataset(get_file_text('datasets_ours/news/train.csv'), tokenizer, classes_dict)
    # val = NewsDataset(get_file_text('datasets_ours/news/dev.csv'), tokenizer, classes_dict)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # record the eval results
    labels_all = []
    predictions_all = []

    fold = 1
    for train_ids, test_ids in kfold.split(train):
        print(f'FOLD {fold}')
        print('----------------------')

        # init the fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(train, batch_size=1, sampler=test_subsampler)

        # fresh model
        model = torch.load(BASE_MODEL_PATH)

        # optimization
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = transformers.AdamW(model.parameters(), lr=learning_rate, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        sigmoid = torch.nn.Sigmoid()

        # metrics
        train_metric = torchmetrics.F1Score().to(device)
        # writer = SummaryWriter(output_dir + '/logs')

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
            for train_input, train_label in tqdm.tqdm(trainloader):
                train_label = train_label.to(device)
                mask = torch.squeeze(train_input[1].to(device), dim=0)
                input_id = train_input[0].squeeze(1).to(device)

                output = model(input_id, mask).logits

                with torch.autocast('cuda'):
                    batch_loss = criterion(output, train_label)

                # writer.add_scalar('loss/train', float(batch_loss), epoch_num * len(train) + iteration)
                train_metric(sigmoid(output), torch.tensor(train_label, dtype=torch.int32))

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                iteration += 1

            # writer.add_scalar('accuracy/train', train_metric.compute(), (epoch_num + 1) * len(train))
            print(f'F1 TRAIN: {float(train_metric.compute())}')
            train_metric.reset()
            model.eval()

            # eval only on the last epoch
            if epoch_num < epochs - 1:
                continue

            for val_input, val_label in tqdm.tqdm(testloader):
                val_label = torch.tensor(torch.unsqueeze(val_label, dim=-1), dtype=torch.int)
                val_label = val_label.to(device)

                mask = val_input[1].to(device)
                input_id = val_input[0].squeeze(1).to(device)

                output = model(input_id, mask).logits
                output = sigmoid(output)
                predictions_all.append(output.to('cpu'))
                labels_all.append(torch.tensor(val_label, dtype=torch.int32).to('cpu'))

            fold += 1

    eval_metric = torchmetrics.F1Score().to('cpu')
    for prediction, label in zip(predictions_all, labels_all):
        eval_metric(prediction, label)
    print('------------------------------------')
    print(f'F1 Score: {float(eval_metric.compute())}')


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

BASE_MODEL_PATH = 'basemodel'

try:
    os.mkdir(output_dir)
except OSError:
    pass

classes_dict = get_class_dict()
model = transformers.BertForSequenceClassification.from_pretrained(model_name, num_labels=len(classes_dict), from_tf=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

torch.save(model, BASE_MODEL_PATH)

train(LR, EPOCHS)
