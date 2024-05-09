import json
import os
import re
from tqdm import tqdm

uris = sorted(os.listdir("chat"),
              key=lambda s: int(re.compile(r'.*_(.*)\.json').match(s).group(1)))

# Filter predicate. A message with by `author` with `content` is
# included in the dataset iff `pred(author, content) == True`.
def pred(author: str, content: str) -> bool:
    return '.discordapp.' not in content

lines: list[str] = []

# chat = ''
for uri in tqdm(uris):
    with open('chat/' + uri, 'r') as f:
        data = json.loads(f.read())
        for o in data:
            if o['content'] and pred(o['author']['username'], o['content']):
                lines.append(f"{o['author']['username']}: {o['content']}")

lines.reverse()
chat = '\n'.join(lines)

def sanitize(s: str) -> str:
    m = { 'raskolnikov9655': 'raskolnikov',
          # 'lee9282': 'lee',
          # 'ks8169': 'ks',
          # 'intptr_t': 'Six380',
    }
    for k, v in m.items():
        s = s.replace(k, v)
    return s

chat = sanitize(chat)

with open('out.txt', 'w') as f:
    f.write(chat)
