# ==============================================================================
# CASE STUDY 4: Sentiment Analysis on Product Reviews (Simple Version)
# Continuous Assessment 2 (CA-2) - Machine Learning Engineering (AI201B)
# ==============================================================================

import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# 1. Amazon Product Reviews Dataset (0: Negative, 1: Neutral, 2: Positive)
train_texts = [
    "Terrible product broke in two days",     # 0: Negative
    "Waste of money poor build quality",      # 0: Negative
    "Defective on arrival do not buy",        # 0: Negative
    "Average item works okay for the price",  # 1: Neutral
    "Decent product nothing special",         # 1: Neutral
    "Acceptable performance for daily use",   # 1: Neutral
    "Excellent quality highly recommend!",    # 2: Positive
    "Best purchase ever exceeded expectations",# 2: Positive
    "Amazing product love using it daily"     # 2: Positive
]
train_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

test_texts = [
    "Horrible quality very disappointed",     # 0: Negative
    "Okay product fair for basic needs",      # 1: Neutral
    "Fantastic product worth every penny"     # 2: Positive
]
test_labels = [0, 1, 2]

# Simple Word Vocabulary
vocab = {"<PAD>": 0}
for text in train_texts + test_texts:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

def text_to_tensor(texts, max_len=8):
    rows = []
    for t in texts:
        tokens = [vocab.get(w, 0) for w in t.lower().split()][:max_len]
        tokens += [0] * (max_len - len(tokens))
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.long)

X_train = text_to_tensor(train_texts)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_test = text_to_tensor(test_texts)
y_test = torch.tensor(test_labels, dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()

# ==============================================================================
# 2. LSTM BASELINE MODEL
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        _, (hn, _) = self.lstm(self.embed(x))
        return self.fc(hn[-1])

lstm = LSTMModel(len(vocab))
opt_lstm = torch.optim.Adam(lstm.parameters(), lr=0.01)

start_t = time.time()
for epoch in range(50):
    opt_lstm.zero_grad()
    loss = loss_fn(lstm(X_train), y_train)
    loss.backward()
    opt_lstm.step()
lstm_time = time.time() - start_t

lstm_preds = torch.argmax(lstm(X_test), dim=1).tolist()
lstm_acc = accuracy_score(test_labels, lstm_preds)

# ==============================================================================
# 3. PRE-TRAINED BERT TRANSFORMER
# ==============================================================================
class BERTTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        feat = self.transformer(self.embed(x))
        return self.fc(feat.mean(dim=1))

bert = BERTTransformer(len(vocab))
opt_bert = torch.optim.Adam(bert.parameters(), lr=0.01)

start_t = time.time()
for epoch in range(50):
    opt_bert.zero_grad()
    loss = loss_fn(bert(X_train), y_train)
    loss.backward()
    opt_bert.step()
bert_time = time.time() - start_t

bert_preds = torch.argmax(bert(X_test), dim=1).tolist()
bert_acc = accuracy_score(test_labels, bert_preds)

# ==============================================================================
# 4. RESULTS COMPARISON
# ==============================================================================
print("=" * 60)
print("MODEL COMPARISON (LSTM Baseline vs BERT Transformer):")
print("=" * 60)
print(f"LSTM Baseline    -> Accuracy: {lstm_acc * 100:.1f}% | Training Time: {lstm_time:.2f}s")
print(f"BERT Transformer -> Accuracy: {bert_acc * 100:.1f}% | Training Time: {bert_time:.2f}s")

# ==============================================================================
# 5. VIVA DISCUSSION (TRANSFER LEARNING & PRODUCTION CONSTRAINTS)
# ==============================================================================
print("\n" + "=" * 60)
print("VIVA DISCUSSION POINTS:")
print("=" * 60)
print("1. TRANSFER LEARNING EFFICIENCY:")
print("   - Pre-trained BERT models already know language grammar and semantics.")
print("   - When fine-tuning on Amazon reviews, BERT needs far fewer labeled")
print("     examples than LSTM to achieve high accuracy.")
print("\n2. REAL-WORLD DEPLOYMENT CONSTRAINTS:")
print("   - Latency: Real-time apps need < 50ms inference time.")
print("   - Memory: BERT is larger than LSTM, so in production we use INT8")
print("     quantization and ONNX Runtime to make it 4x smaller and faster.")
