"""单头自注意力（M2 第一块积木）。

输入  x   [B, T, C]
输出  out [B, T, head_dim]，以及注意力权重 [B, T, T]

核心三步：
  1. 每个 token 生成自己的 q / k / v
  2. scores = q @ k^T / sqrt(head_dim)  —— 每个位置对每个位置打分
  3. 因果遮罩 + softmax（按行归一化）→ 用权重把 v 加权求和
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, n_embd: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        # 三个投影：没有 bias，和 GPT-2 一致
        self.wq = nn.Linear(n_embd, head_dim, bias=False)
        self.wk = nn.Linear(n_embd, head_dim, bias=False)
        self.wv = nn.Linear(n_embd, head_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True):
        B, T, C = x.shape

        q = self.wq(x)                                  # [B, T, hd]
        k = self.wk(x)                                  # [B, T, hd]
        v = self.wv(x)                                  # [B, T, hd]

        # [B,T,hd] @ [B,hd,T] -> [B,T,T]：第 i 行第 j 列 = 位置 i 对位置 j 的原始分数
        scores = (q @ k.transpose(-2, -1)) * self.scale

        if causal:
            # 上三角（不含对角线）置 -inf：位置 i 只能看 j <= i
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)              # 每一行加起来 = 1
        out = weights @ v                                # [B,T,T] @ [B,T,hd] -> [B,T,hd]
        return out, weights
