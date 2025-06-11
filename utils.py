def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().replace("?", "").replace("'", " ").split()

def build_vocab(df):
    vocab = {'<UNK>': 0}
    for _, row in df.iterrows():
        tokens = tokenize(row['question']) + tokenize(row['answer'])
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokenize(text)]
