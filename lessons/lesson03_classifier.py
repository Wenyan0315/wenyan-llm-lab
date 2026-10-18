# Lesson 03：从零手写一个 2 类线性分类器（不用 nn.Linear、不用 optimizer）
# 运行：cd wenyan-llm-lab && .venv/bin/python lessons/lesson03_classifier.py
#
# 串联前两课：
#   Lesson 01 的交叉熵（这次 100 个样本一起算）
#   Lesson 02 的三步舞（这次更新公式自己写，不用 optimizer.step）

import torch

torch.manual_seed(0)

# ==== 1. 造数据：两团 2 维点 ====
# 第 0 类：中心 (0,0) 附近；第 1 类：中心 (3,3) 附近。各 50 个点。
n = 50
class0 = torch.randn(n, 2)                                # [50, 2] 围绕 (0,0) 散布
class1 = torch.randn(n, 2) + torch.tensor([3.0, 3.0])     # [50, 2] 围绕 (3,3) 散布
x = torch.cat([class0, class1])                           # [100, 2] 全部输入
y = torch.cat([torch.zeros(n, dtype=torch.long),          # [100] 标准答案：
               torch.ones(n, dtype=torch.long)])          #   前 50 个是第 0 类，后 50 个是第 1 类

# ==== 2. 手工建参数 ====
# 模型：logits = x @ w + b
# w 形状 [2, 2]：输入 2 维 -> 输出 2 个类别的分数
#   可以把 w 的每一列理解成"一个类别专属的打分器"
w = torch.randn(2, 2, requires_grad=True)   # requires_grad=True：告诉 autograd 要对它求梯度
b = torch.zeros(2, requires_grad=True)

print(f"初始 w =\n{w}\n初始 b = {b.tolist()}\n")

# ==== 3. 训练循环：三步舞的裸写版 ====
lr = 0.1
for step in range(50):
    logits = x @ w + b      # [100,2] @ [2,2] -> [100,2]，再 + [2] 广播（Lesson 01 的两条规则！）
    loss = torch.nn.functional.cross_entropy(logits, y)
    # cross_entropy(logits[N,C], y[N])：N 个样本各自算 -ln(p[正确])，再取平均
    # —— 和 Lesson 01 的 loss_fn(logits, torch.tensor(2)) 是同一个函数，只是批量版

    pred = logits.argmax(dim=1)     # [100] 每个样本取分数最大的类别作为模型答案
    acc = (pred == y).float().mean()  # 答对的比例

    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  loss = {loss.item():.4f}  acc = {acc.item():.2f}")

    # ---- 三步舞（裸写版）----
    loss.backward()                 # 摸坡：算出 w.grad 和 b.grad，参数不动
    with torch.no_grad():           # 更新参数不是计算图的一部分，关掉追踪
        w -= lr * w.grad            # <- optimizer.step() 内部干的就是这两行
        b -= lr * b.grad
        w.grad.zero_()              # <- optimizer.zero_grad() 内部干的就是这两行
        b.grad.zero_()

# ==== 4. 考 3 个没见过的点 ====
# 前两个明显属于某一类，第三个 (1.5, 1.5) 落在两团正中间——边界地带
test = torch.tensor([[0.1, -0.2], [2.8, 3.5], [1.5, 1.5]])
with torch.no_grad():
    t_logits = test @ w + b         # [3, 2]
    t_p = torch.softmax(t_logits, dim=1)   # [3, 2] 每行加起来等于 1
print()
for pt, lg, pr in zip(test.tolist(), t_logits.tolist(), t_p.tolist()):
    ans = 0 if lg[0] > lg[1] else 1
    print(f"点 {pt} -> 判为第 {ans} 类，概率 [{pr[0]:.3f}, {pr[1]:.3f}]")

# ==== 5. 你的亲手实验（改之前先预测）====
#
# 实验 1：把 class1 的中心从 [3.0, 3.0] 改成 [1.0, 1.0]（两团挨得更近）
#         预测：acc 还能到 1.00 吗？边界点 (1.5,1.5) 的概率会怎么变？
#
# 实验 2：把 lr 改成 10.0，预测会不会像 Lesson 02 那样发散。
#
# 实验 3（选做）：把 n 改成 5（每类只有 5 个点），预测训练后 acc 会更不稳定吗？
