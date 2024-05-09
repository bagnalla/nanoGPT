"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer

dataset = 'flatearth'
vocab_size = 1536

# open some text and train a vocab of vocab_size tokens
text = open(f'/home/alex/data/gpt/data/{dataset}/chat.txt', 'r',
            encoding='utf-8').read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("token", exist_ok=True)

t0 = time.time()
# for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
for TokenizerClass, name in zip([RegexTokenizer], ["regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, vocab_size, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join(f'/home/alex/data/gpt/models/{dataset}', name + str(vocab_size))
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
