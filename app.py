import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import types

# Patch torch.classes for Streamlit compatibility
if not isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

# Page config
st.set_page_config(page_title="India QA System", page_icon="🧠", layout="centered")

st.title("General Knowledge QA System")
st.markdown("Upload a **CSV file** with `question, answer` columns. Ask a question and get a one-word answer.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV (Max 2MB)", type=["csv"], key="file_upload")

# Enforce size limit
if uploaded_file and uploaded_file.size > 2 * 1024 * 1024:
    st.error("❌ File too large. Please upload a file smaller than 2MB.")
    st.stop()

# ---------------------------
# Utilities
# ---------------------------
def tokenize(text):
    if not isinstance(text, str):
        text = ''
    text = text.lower().replace("?", "").replace("'", " ")
    return text.split()

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokenize(text)]

class QADataset(Dataset):
    def __init__(self, df, vocab):
        self.data = []
        for _, row in df.iterrows():
            q = text_to_indices(row["question"], vocab)
            a = text_to_indices(row["answer"], vocab)
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
        _, hidden = self.rnn(x)
        return self.fc(hidden.squeeze(0))

def train_model(df, vocab):
    dataset = QADataset(df, vocab)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = SimpleRNN(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    for epoch in range(10):
        for question, answer in dataloader:
            optimizer.zero_grad()
            output = model(question)
            target_token = answer[0][0].unsqueeze(0)
            loss = criterion(output, target_token)
            loss.backward()
            optimizer.step()

    return model

def build_vocab(df):
    vocab = {'<UNK>': 0}
    for _, row in df.iterrows():
        for token in tokenize(row["question"]) + tokenize(row["answer"]):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def predict(model, question, vocab):
    model.eval()
    indices = text_to_indices(question, vocab)
    if not indices:
        return "Invalid input"
    tensor = torch.tensor(indices).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, pred_idx = torch.max(probs, dim=1)
        inv_vocab = {v: k for k, v in vocab.items()}
        return inv_vocab.get(pred_idx.item(), "Unknown")

# ---------------------------
# Training Step (Only once)
# ---------------------------
if uploaded_file and "model" not in st.session_state:
    df = pd.read_csv(uploaded_file)
    if 'question' not in df.columns or 'answer' not in df.columns:
        st.error("CSV must contain 'question' and 'answer' columns.")
        st.stop()

    with st.spinner("Training model... ⏳"):
        vocab = build_vocab(df)
        model = train_model(df, vocab)
        st.session_state.model = model
        st.session_state.vocab = vocab
    st.success("Model trained successfully ✅")

# ---------------------------
# Prediction Step
# ---------------------------
if "model" in st.session_state:
    question_input = st.text_input("Ask a question:")
    if question_input:
        answer = predict(st.session_state.model, question_input, st.session_state.vocab)
        st.markdown(f"**Answer:** {answer}")
