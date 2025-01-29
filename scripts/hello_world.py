import torch
import torch.nn.functional as F

vocab = {
    "?": 61,
    "Hello": 72,
    "my": 44,
    "name": 21,
    "is": 93,
    "Bes": 11,
}

sentence = ["?", "my", "?", "is", "Bes"]

class Magic(torch.nn.Module):
    def __init__(self):
        super(Magic, self).__init__()

    def forward(self, inputs):
        pass


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.emb = torch.nn.Embedding(128, 9)
        self.magics = torch.nn.ModuleList([Magic() for _ in range(3)])
        self.linear = torch.nn.Linear(9, 128)

    def forward(self, inputs):
        embs = self.emb(inputs)
        for magic in self.magics:
            embs = magic(embs)
        probs = F.log_softmax(embs, dim=1)
        return probs

print('hello world')
