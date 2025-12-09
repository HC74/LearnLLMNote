import torch
import torch.nn as nn
from torch.nn import functional as F
import math

d_model = 512
context_length = 16
num_heads = 8
batch_size = 4
# 64
head_size = d_model // num_heads
dropout = 0.1
num_blocks = 12
max_token_value = 100256 #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 前馈网络
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            # 变大 4倍
            nn.Linear(d_model, 4 * d_model),
            # 激活
            nn.ReLU(),
            # 变小 4倍
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)

# 单头
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_size, bias=False)
        self.Wk = nn.Linear(d_model, head_size, bias=False)
        self.Wv = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        self.Dropout = nn.Dropout(dropout)

    # x 样本文字
    def forward(self, x):  # x: [batch_size , [Timestep],context_length, head_size]
        B, T, D = x.shape
        # q: [batch_size,context_length, head_size]
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        output = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        output = output.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        output = F.softmax(output, dim=-1)
        # output = self.Dropout(output)  # Optional

        output = output @ v
        return output


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 8头
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.Dropout(self.Wo(output))

        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_linear = nn.Linear(d_model,max_token_value)
        self.te_lookup_table = nn.Embedding(max_token_value, d_model)
        self.transformer_block = nn.Sequential(
            *[TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)]
        )


    def forward(self, x_batch,y_batch=None):
        B,T = x_batch.shape
        pe_lookup_table = torch.zeros(context_length,d_model,device = device)
        posistion = torch.arange(0,context_length,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0,d_model,2).float() / d_model)

        pe_lookup_table[:, 0::2] = torch.sin(posistion * div_term)
        pe_lookup_table[:, 1::2] = torch.cos(posistion * div_term)

        pe = pe_lookup_table[:T,:]
        output = self.te_lookup_table(x_batch) + pe
        output = self.transformer_block(output)
        logit = self.vocab_linear(output)

        if y_batch is not None:
            B,T,D = logit.shape
            logit_reshaped = logit.view(B * T, D)
            y_reshaped = y_batch.view(B * T)

            loss = F.cross_entropy(input=logit_reshaped, target=y_reshaped)
        else:
            loss = None

        return logit, loss

    def generate(self, x_batch,max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            # x_batch [batch_size, context_length(Timestep), ]
            x_crop = x_batch[:, -context_length:]
            logits, _ = self.forward(x_crop) # [batch_size, context_length(timestep), vocab_size]
            logits = logits[:, -1, :] / temperature
            proabilities = F.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(proabilities, num_samples=1)
            x_batch = torch.cat((x_batch, predicted_token), dim=1)

        return x_batch


