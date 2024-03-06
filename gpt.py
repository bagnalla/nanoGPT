import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All the unique characters that occur in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map characters to integers.
stoi = { ch:i for i,ch in enumerate(chars) }

# Map integers to characters.
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ixs: list[int]) -> str:
    return ''.join([itos[i] for i in ixs])

s = 'hello this is a string'
# print(s == decode(encode(s)))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)

# Split up the data into train and validation sets.
n = int(0.9*len(data)) # first 90% will be train, rest val.
train_data = data[:n]
val_data = data[n:]

batch_size = 4
block_size = 8

# Each context sequence in x will be used as 8 training examples, one
# for each partial prefix of the context (up to and including the
# entire context). That's why y has the same shape as x, because for
# each context sequence there are 8 target values for the expected
# next character given the context up to that point.
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[ix:ix+block_size] for ix in ixs])
    y = torch.stack([data[ix+1:ix+block_size+1] for ix in ixs])
    return x, y

# This helps to understand how the batches are being built (in
# particular, the intended correspondence between x and y).
torch.manual_seed(1337)
xb, yb = get_batch('train')
# print(xb.shape, yb.shape)
# print('----')
# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f'when input is {context.tolist()} the target is {target}')

torch.manual_seed(1337)

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Each token directly reads off the logits for the next token
        # from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.token_embedding_table(x) # (B, T, C)

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

    def generate(self, ixs, max_new_tokens):
        # ixs is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            logits, _ = self(ixs)
            # Focus only on the last time step.
            logits = logits[:, -1, :] # Becomes (B, C).
            # Get probabilities.
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Draw sample.
            ix_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence.
            ixs = torch.cat([ixs, ix_next], dim=1) # (B, T+1)
        return ixs

m = BigramModel(vocab_size)
out, loss = m(xb, yb)
print(out.shape)
print(loss.item())

print(decode(m.generate(ixs=torch.zeros((1, 1), dtype=torch.long),
                        max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for _ in range(100000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(ixs=torch.zeros((1, 1), dtype=torch.long),
                        max_new_tokens=400)[0].tolist()))
