import torch
import tiktoken
from model import Model

batch_size = 12
context_length = 16
max_iters = 200
learning_rate = 1e-3  # 0.001
evel_interval = 20
evel_iters = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

with open('commodity.csv', 'r',encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('cl100k_base')
tokenized_text = tokenizer.encode(text)
tokenized_text = torch.tensor(data=tokenized_text, dtype=torch.long, device=device)

p_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:p_size]
vaild_data = tokenized_text[p_size:]

model = Model().to(device)


def get_batch(split):
    data = train_data if split == 'train' else vaild_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + 1 + context_length] for idx in idxs])
    return x, y


def estimate_loss():
    out = {}
    losses = torch.zeros(evel_iters)
    for split in ['train', 'valid']:
        for k in range(evel_iters):
            x, y = get_batch('train')
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % evel_interval == 0 or step == max_iters - 1:
        loss = estimate_loss()
        print('Step', step, 'Trains loss', round(loss['train'].item(),3),'Valids loss', round(loss['valid'].item(),3))

    x, y = get_batch('train')
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.ckpt')
