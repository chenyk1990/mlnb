import torch
import torch.nn as nn
import torch.optim as optim


## define self-attention module
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, seq_length, d_model = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.reshape(N, seq_length, self.num_heads, self.d_k)
        K = K.reshape(N, seq_length, self.num_heads, self.d_k)
        V = V.reshape(N, seq_length, self.num_heads, self.d_k)

        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        attention = torch.softmax(energy / (self.d_k ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, V])
        out = out.reshape(N, seq_length, d_model)

        return self.fc_out(out)


## Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(attn_output + x)
        ff_output = self.ff(x)
        x = self.norm2(ff_output + x)
        return self.dropout(x)

## Define transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
        for _ in range(num_layers)])
        self.embed = nn.Linear(input_dim, d_model)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x



## toy example
data = torch.rand(10, 5, 8)  # (batch_size, seq_length, input_dim)

model = TransformerEncoder(input_dim=8, d_model=32, num_layers=2, num_heads=4, dropout=0.1)

output = model(data)
print(output.shape)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Another toy example
d_model = 8
seq_length = 5
num_heads = 2

# Query, Key, Value
np.random.seed(42)
Q = np.random.rand(seq_length, d_model)
K = np.random.rand(seq_length, d_model)
V = np.random.rand(seq_length, d_model)

# energy (dot product)
energy = np.dot(Q, K.T)

# self-attention weight
attention_weights = np.exp(energy) / np.sum(np.exp(energy), axis=1, keepdims=True)

# SA weight plot
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, annot=True, cmap="viridis", xticklabels=range(seq_length), yticklabels=range(seq_length))
plt.title("Self-Attention Weights")
plt.xlabel("Key Positions")
plt.ylabel("Query Positions")
plt.show()


# Create multi-head example
num_heads = 4
d_k = d_model // num_heads

# create multi-head Query, Key, Value
Q_heads = np.random.rand(seq_length, num_heads, d_k)
K_heads = np.random.rand(seq_length, num_heads, d_k)
V_heads = np.random.rand(seq_length, num_heads, d_k)

attention_heads = []
for i in range(num_heads):
    energy_head = np.dot(Q_heads[:, i, :], K_heads[:, i, :].T)
    attention_head = np.exp(energy_head) / np.sum(np.exp(energy_head), axis=1, keepdims=True)
    attention_heads.append(attention_head)

# SA weight for each head
fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
for i, attention_head in enumerate(attention_heads):
    sns.heatmap(attention_head, annot=True, cmap="viridis", ax=axes[i], xticklabels=range(seq_length), yticklabels=range(seq_length))
    axes[i].set_title(f"Head {i + 1}")

plt.suptitle("Multi-Head Attention Weights")
plt.show()


# create data
x = np.random.rand(seq_length, d_model)

# attention output
attn_output = np.random.rand(seq_length, d_model)

# feed-forward output
ff_output = np.random.rand(seq_length, d_model)

# visualize how Transformer Block works
plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
sns.heatmap(x, annot=True, cmap="Blues")
plt.title("Input Sequence")

plt.subplot(1, 3, 2)
sns.heatmap(attn_output, annot=True, cmap="Greens")
plt.title("Self-Attention Output")

plt.subplot(1, 3, 3)
sns.heatmap(ff_output, annot=True, cmap="Reds")
plt.title("Feed-Forward Output")

plt.suptitle("Transformer Block Processing")
plt.show()


# create data
x = np.random.rand(seq_length, d_model)
num_layers = 3

# output of each layer
layer_outputs = [np.random.rand(seq_length, d_model) for _ in range(num_layers)]

# visualize how Transformer Encoder works
fig, axes = plt.subplots(1, num_layers + 1, figsize=(18, 6))
sns.heatmap(x, annot=True, cmap="Blues", ax=axes[0])
axes[0].set_title("Input Sequence")

for i, layer_output in enumerate(layer_outputs):
    sns.heatmap(layer_output, annot=True, cmap="Purples", ax=axes[i + 1])
    axes[i + 1].set_title(f"Layer {i + 1} Output")

plt.suptitle("Transformer Encoder Layers")
plt.show()





































