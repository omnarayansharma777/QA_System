import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import text_to_indices, tokenize

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.data = []
        for _, row in df.iterrows():
            q = text_to_indices(row['question'], vocab)
            a = text_to_indices(row['answer'], vocab)
            if q and a:
                self.data.append((q, a))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        return torch.tensor(q), torch.tensor(a)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        final = output[:, -1, :]
        return self.fc(final)

def train_model(df, vocab):
    dataset = QADataset(df, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SimpleRNN(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for question, answer in dataloader:
            optimizer.zero_grad()
            output = model(question)
            target = answer[0][0].unsqueeze(0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

def predict(model, vocab, question):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = text_to_indices(question, vocab)
    if not tokens:
        return "I don't understand."
    input_tensor = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        _, index = torch.max(probs, dim=1)
    return inv_vocab.get(index.item(), "<UNK>")
