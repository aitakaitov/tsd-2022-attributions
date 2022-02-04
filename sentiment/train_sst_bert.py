from torch.optim import Adam
from tqdm import tqdm
import torch
import torchmetrics

from sst_dataset import SSTDataset
from imdb_dataset import IMDbDataset
from bert_512 import BERT512


def train(model, learning_rate, epochs):
    dataset = IMDbDataset()
    train, val = torch.utils.data.random_split(dataset, [400, 100], torch.Generator().manual_seed(0))

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_metric = torchmetrics.Accuracy().to(device)
    val_metric = torchmetrics.Accuracy().to(device)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        for train_input, train_label in tqdm(train_dataloader):
            train_label = torch.unsqueeze(train_label, dim=-1)
            train_label = train_label.to(device)
            mask = train_input[1].to(device)
            input_id = train_input[0].squeeze(1).to(device)

            output = model(input_id, mask)

            with torch.autocast('cuda'):
                batch_loss = criterion(output, train_label)

            train_metric(output, torch.tensor(train_label, dtype=torch.int32))

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch_num} train accuracy: {train_metric.compute()}')
        train_metric.reset()

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = torch.unsqueeze(val_label, dim=-1)
                val_label = val_label.to(device)

                mask = val_input[1].to(device)
                input_id = val_input[0].squeeze(1).to(device)

                output = model(input_id, mask)
                val_metric(output, torch.tensor(val_label, dtype=torch.int32))

        print(f'Epoch {epoch_num} validation accuracy: {val_metric.compute()}')
        val_metric.reset()


model = BERT512()
EPOCHS = 10
LR = 0.000001

train(model, LR, EPOCHS)
