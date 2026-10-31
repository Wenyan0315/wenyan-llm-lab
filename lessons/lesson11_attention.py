# Lesson 11：单头 attention —— shape 追踪 + 因果性实验
# 运行：.venv/bin/python lessons/lesson11_attention.py

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 lessons/ 里的脚本能 import src/
from src.tiny_model.attention import SingleHeadAttention

torch.manual_seed(0)

B, T, C, HD = 2, 5, 16, 8
x = torch.randn(B, T, C)
attn = SingleHeadAttention(C, HD)

out, weights = attn(x, causal=True)

print("== shape 追踪 ==")
print(f"x        {tuple(x.shape)}")
print(f"out      {tuple(out.shape)}")
print(f"weights  {tuple(weights.shape)}")

print("\n== 第 0 个样本的注意力矩阵（行=谁在看，列=被看的是谁）==")
print(weights[0].detach().round(decimals=3))
print("每一行之和：", weights[0].sum(dim=-1).detach().round(decimals=4).tolist())

print("\n== 因果性检查 ==")
print("因果遮罩下，第 0 个 token 只能看自己 → 权重应为 [1, 0, 0, 0, 0]")
print("实测：", weights[0, 0].detach().round(decimals=3).tolist())

# 改掉最后一个 token，再看前面位置的输出会不会变
x2 = x.clone()
x2[:, -1, :] = torch.randn(B, C)

out_a, _ = attn(x, causal=True)
out_b, _ = attn(x2, causal=True)
diff = (out_a - out_b).abs()
print(f"\n因果模式：改第 4 个 token 后，各位置输出变化量 {diff[0].max(dim=-1).values.tolist()}")

out_a2, _ = attn(x, causal=False)
out_b2, _ = attn(x2, causal=False)
diff2 = (out_a2 - out_b2).abs()
print(f"无遮罩  ：改第 4 个 token 后，各位置输出变化量 {diff2[0].max(dim=-1).values.tolist()}")

print("\n== 与 PyTorch 参考实现对照 ==")
q = attn.wq(x)
k = attn.wk(x)
v = attn.wv(x)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print("最大绝对误差：", (out - ref).abs().max().item())
