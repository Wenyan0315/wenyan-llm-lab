# Lesson 07：把 shifted labels 用在真实文本上（含 chunk 切法）
# 目标：亲眼看到 x / y 是怎么错位一格的，以及长序列怎么切成固定长度的块
# 运行：.venv/bin/python lessons/lesson07_shift_labels.py

import torch

TEXT = (
    "the cat sat on the mat. "
    "the dog sat on the log. "
    "the cat and the dog sat on the mat."
)

# ── 1. 编码：字符级（复用 Lesson 04 的做法，表 = 这段文本里出现过的所有字符）
chars = sorted(set(TEXT))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}
tokens = [char_to_id[c] for c in TEXT]
print(f"原文字符数 {len(TEXT)}   token 数 {len(tokens)}   词表大小 {len(chars)}")

# ── 2. 错位一格：x 是"读到的部分"，y 是"要预测的下一个"
x = tokens[:-1]                 # 从头到倒数第二个
y = tokens[1:]                  # 从第二个到最后一个
print(f"\nx 长度 {len(x)}   y 长度 {len(y)}")

# 上下两行错开一格，一眼就能看出错位
print("\nx（输入）：", "".join(id_to_char[t] for t in x[:24]))
print("y（目标）：", "".join(id_to_char[t] for t in y[:24]))
print("对应数字 x：", x[:24])
print("对应数字 y：", y[:24])

# 硬验证：x 的第 i 个位置的正确答案，永远是原文的 tokens[i+1]
assert all(y[i] == tokens[i + 1] for i in range(len(x)))
print("\n对齐验证通过：x[i] 的答案永远 = tokens[i+1]")

# ── 3. 切 chunk：整条序列喂不进去，要切成固定长度的块
BLOCK = 12                          # 每块让模型看 12 个 token，并预测 12 个
STEP = BLOCK + 1                    # 但实际要吃 13 个 token（错位共享一个）
n_chunks = len(tokens) // STEP
print(f"\n每块吃 {STEP} 个 token → 能切出 {n_chunks} 块，"
      f"多出来的 {len(tokens) - n_chunks * STEP} 个 token 被丢掉")

for k in range(n_chunks):
    seg = tokens[k * STEP:(k + 1) * STEP]
    cx, cy = seg[:-1], seg[1:]      # 同一段，两种切法
    print(f"  块{k}  x={''.join(id_to_char[t] for t in cx)!r}")
    print(f"        y={''.join(id_to_char[t] for t in cy)!r}")

# ── 4. 叠成 batch：这才是模型真正吃到的东西
xs = torch.tensor([tokens[k * STEP:k * STEP + BLOCK] for k in range(n_chunks)])
ys = torch.tensor([tokens[k * STEP + 1:(k + 1) * STEP] for k in range(n_chunks)])
print(f"\nbatch 形状：x = {tuple(xs.shape)}   y = {tuple(ys.shape)}")
print("▲ 这就是 M0 讲过的 [B, T]：B = batch 里几条样本，T = 每条看多少个 token")
print("▲ logits 会是 [B, T, V]：每个位置对全词表各出一个分数，其中只有 y 那一个是对的")
