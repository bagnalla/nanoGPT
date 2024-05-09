# With self-attention.

import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from minbpe import RegexTokenizer

# Hyperparameters
batch_size = 64
block_size = 256 # Context length
max_iters = 300000
eval_interval = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # Self-attention embedding dimensionality.
n_head = 6   # Number of heads. Each head is therefore 384/6 = 64 dimensions.
n_layer = 6
dropout = 0.2
vocab_size = 768

# learning_rate = 1e-4
learning_rate = 1e-4
lr_alpha = 1.0
lr_interval = 1000000

dataset = 'wq'
gen_n = 5000 # Number of tokens to generate at inference time.

print(device)

text = open(f'/home/alex/data/gpt/data/{dataset}/chat.txt', 'r',
            encoding='utf-8').read()

tokenizer = RegexTokenizer()
tokenizer.load(f'/home/alex/data/gpt/models/{dataset}/regex{vocab_size}.model')

train_data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[ix:ix+block_size] for ix in ixs])
    y = torch.stack([data[ix+1:ix+block_size+1] for ix in ixs])
    return x, y

class Head(nn.Module):
    """ One head of self-attention. """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores "affinities".
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values.
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel. """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity. """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ asdf """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb    # (B, T, n_embd)
        x = self.blocks(x)       # (B, T, n_embd)
        x = self.ln_f(x)         # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape to form expected by F.cross_entropy. Instead of
            # messing around with the 3D shape (cross_entropy wants (B, C,
            # T)) we just flatten B and T into a single dimension.
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # ixs is (B, T) array of indices in the current context.
        for _ in tqdm(range(max_new_tokens)):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # Focus only on the last time step.
            logits = logits[:, -1, :] # Becomes (B, C).
            # Get probabilities.
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Draw sample.
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence.
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
            # yield idx_next
        return idx

# print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device),
#                             max_new_tokens=100)[0].tolist()))

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    # for split in ['train', 'val']:
    for split in ['train']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model=None, i=0):
    if model is None:
        model = BigramModel().to(device)

    global learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(1, max_iters+1):
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {step}: train loss = {losses['train']:.4f}")
            i += 1
            torch.save(model.state_dict(), f'/home/alex/data/gpt/models/{dataset}/model{i}.pt')
            #         print(f"step {step}: train loss = {losses['train']:.4f}, \
# val loss = {losses['val']:.4f}")

        if step % lr_interval == 0:
            learning_rate = lr_alpha * learning_rate
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
            print(f'{learning_rate=}')

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # i += 1
    # torch.save(model.state_dict(), f'models/model{i}.pt')

if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
elif len(sys.argv) > 2 and sys.argv[1] == 'resume':
        model = BigramModel().to(device)
        i = int(sys.argv[2])
        model.load_state_dict(torch.load(f'/home/alex/data/gpt/models/{dataset}/model{i}.pt'))
        train(model=model, i=i)
else:
    i = int(sys.argv[1])
    model = BigramModel().to(device)
    model.load_state_dict(torch.load(f'/home/alex/data/gpt/models/{dataset}/model{i}.pt'))
    print(tokenizer.decode(model.generate(idx=torch.tensor([tokenizer.encode('\n')],
                                                           dtype=torch.long, device=device),
                                          max_new_tokens=gen_n)[0].tolist()))
