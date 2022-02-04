import torch


def load_model():
    config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', config=config)
    return model


class BERT512(torch.nn.Module):

    def __init__(self):
        super(BERT512, self).__init__()
        self.bert = load_model()
        self.linear1 = torch.nn.Linear(768, 20)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        res = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.linear1(res.pooler_output)
        x = self.activation(x)
        x = self.linear2(x)
        return self.activation(x)
