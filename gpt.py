# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# # Hyperparameters
# batch_size = 32
# block_size = 8
# max_iters = 10000
# eval_interval = 1000
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 32

# print(device)

# torch.manual_seed(1337)

# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# # All the unique characters that occur in the text.
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# # Map characters to integers.
# stoi = { ch:i for i,ch in enumerate(chars) }

# # Map integers to characters.
# itos = { i:ch for i,ch in enumerate(chars) }

# def encode(s: str) -> list[int]:
#     return [stoi[c] for c in s]

# def decode(ixs: list[int]) -> str:
#     return ''.join([itos[i] for i in ixs])

# s = 'hello this is a string'
# # print(s == decode(encode(s)))

# data = torch.tensor(encode(text), dtype=torch.long)
# # print(data.shape, data.dtype)

# # Split up the data into train and validation sets.
# n = int(0.9*len(data)) # first 90% will be train, rest val.
# train_data = data[:n]
# val_data = data[n:]

# # Each context sequence in x will be used as 8 training examples, one
# # for each partial prefix of the context (up to and including the
# # entire context). That's why y has the same shape as x, because for
# # each context sequence there are 8 target values for the expected
# # next character given the context up to that point.
# def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
#     data = train_data if split == 'train' else val_data
#     ixs = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[ix:ix+block_size] for ix in ixs])
#     y = torch.stack([data[ix+1:ix+block_size+1] for ix in ixs])
#     return x.to(device), y.to(device)

# # This helps to understand how the batches are being built (in
# # particular, the intended correspondence between x and y).
# xb, yb = get_batch('train')
# # print(xb.shape, yb.shape)
# # print('----')
# # for b in range(batch_size):
# #     for t in range(block_size):
# #         context = xb[b, :t+1]
# #         target = yb[b, t]
# #         print(f'when input is {context.tolist()} the target is {target}')

# class BigramModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
#         self.position_embedding_table = nn.Embedding(block_size, n_embd)
#         self.lm_head = nn.Linear(n_embd, vocab_size)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape
        
#         tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
#         pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
#         x = tok_emb + pos_emb # (B, T, n_embd)
#         logits = self.lm_head(x) # (B, T, vocab_size)

#         if targets is None:
#             loss = None
#         else:
#             # Reshape to form expected by F.cross_entropy. Instead of
#             # messing around with the 3D shape (cross_entropy wants (B, C,
#             # T)) we just flatten B and T into a single dimension.
#             B, T, C = logits.shape
#             logits = logits.view(B*T, C)
#             targets = targets.view(B*T)
#             loss = F.cross_entropy(logits, targets)
        
#         return logits, loss

#     def generate(self, ixs, max_new_tokens):
#         # ixs is (B, T) array of indices in the current context.
#         for _ in range(max_new_tokens):
#             logits, _ = self(ixs)
#             # print("")
#             # print(logits.shape)
#             # Focus only on the last time step.
#             logits = logits[:, -1, :] # Becomes (B, C).
#             # print(logits.shape)
#             # Get probabilities.
#             probs = F.softmax(logits, dim=-1) # (B, C)
#             # Draw sample.
#             ix_next = torch.multinomial(probs, num_samples=1) # (B, 1)
#             # Append sampled index to the running sequence.
#             ixs = torch.cat([ixs, ix_next], dim=1) # (B, T+1)
#         return ixs

# m = BigramModel().to(device)
# out, loss = m(xb, yb)
# print(out.shape)
# print(loss.item())

# print(decode(m.generate(ixs=torch.zeros((1, 1), dtype=torch.long, device=device),
#                         max_new_tokens=100)[0].tolist()))

# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     m.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             _, loss = m(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     m.train()
#     return out

# optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# for step in range(max_iters):
#     if step % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {step}: train loss = {losses['train']:.4f}, \
# val loss = {losses['val']:.4f}")

#     xb, yb = get_batch('train')
    
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
    
# print(loss.item())

# print(decode(m.generate(ixs=torch.zeros((1, 1),
#                                         dtype=torch.long, device=device),
#                         max_new_tokens=500)[0].tolist()))
