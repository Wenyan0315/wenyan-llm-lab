# Lesson 01 验证脚本：label shift / shape 与广播 / softmax 与交叉熵
# 运行前先在纸上写下 4 个预测，再运行核对：
#   cd wenyan-llm-lab && .venv/bin/python lessons/lesson01_tensor_loss.py

import torch

print("=" * 52)
print("第 1 部分：label shift（题 1 验证）")
print("=" * 52)
tokens = [5, 9, 2, 7]
x = tokens[:-1]  # 输入：去掉最后一个（它没有"下一个"可预测）
y = tokens[1:]   # 目标：去掉第一个（它轮不到被预测）
print("输入 x =", x)
print("目标 y =", y)
print("检查：x[i] 之后紧跟的是 y[i] 吗？", all(y[i] == tokens[i + 1] for i in range(len(x))))

print()
print("=" * 52)
print("第 2 部分：shape 与广播（题 2 验证）")
print("=" * 52)
B, T, V = 2, 3, 5
h = torch.randn(B, T, V)  # logits：2 个样本 × 3 个位置 × 5 个候选 token 的分数
w = torch.randn(V)        # 一个 5 维向量
M = torch.randn(V, 8)     # 一个 [5, 8] 矩阵
z = h + w
m = h @ M
print("h.shape =", tuple(h.shape))
print("w.shape =", tuple(w.shape))
print("z = h + w  ->  z.shape =", tuple(z.shape))  # 预测 1：广播后 shape 是什么？
print("m = h @ M  ->  m.shape =", tuple(m.shape))  # 预测 2：哪一维被消耗了？哪一维保留？

print()
print("=" * 52)
print("第 3 部分：softmax 与交叉熵（题 3 验证）")
print("=" * 52)
logits = torch.tensor([2.0, 0.0, -1.0])
p = torch.softmax(logits, dim=-1)
print("logits =", logits.tolist())
print("softmax 概率 p =", [round(v, 4) for v in p.tolist()])  # 预测 3：三个概率大约多少？

loss_fn = torch.nn.CrossEntropyLoss()
# CrossEntropyLoss(logits, 正确类别的下标)：内部自动做 softmax + 交叉熵
loss_case1 = loss_fn(logits, torch.tensor(0))  # 正确 = 第 1 类（下标 0，logit=2.0）
loss_case3 = loss_fn(logits, torch.tensor(2))  # 正确 = 第 3 类（下标 2，logit=-1.0）
print("正确=第 1 类时 loss =", round(loss_case1.item(), 4))  # 预测 4：哪个大？
print("正确=第 3 类时 loss =", round(loss_case3.item(), 4))

print()
print("=" * 52)
print("第 4 部分：亲手实验（改这里）")
print("=" * 52)
# 任务：正确类别固定为下标 2（第三个数）。
# 现在把 my_logits 改成你认为能让 loss < 0.3 的值。
# 运行前先预测：你要改哪个数、往哪个方向改？为什么这会让 loss 变小？
my_logits = torch.tensor([2.0, 0.0, -1.0])  # <- 只改这一行
my_loss = loss_fn(my_logits, torch.tensor(0))
print("my_logits =", my_logits.tolist())
print("my_loss   =", round(my_loss.item(), 4), "（目标 < 0.3）")
