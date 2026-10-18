# Lesson 02：autograd 拟合直线 —— backward / step / zero_grad 三步舞
# 运行：cd wenyan-llm-lab && .venv/bin/python lessons/lesson02_autograd.py
#
# 任务：模型要从 4 个数据点学出 y = 2x + 1（真实规律它事先不知道）
# 跑之前先写下 3 个预测（见课程消息），再运行对照。

import torch

torch.manual_seed(0)  # 固定随机种子，保证每次运行初始值相同（可复现）

# ---- 数据：真实规律 y = 2x + 1 ----
x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])  # shape [4, 1]：4 个样本，1 个特征
y = 2 * x + 1                                    # shape [4, 1]：目标值

# ---- 模型：一个线性层，只有 2 个参数 w 和 b ----
# pred = w * x + b，w、b 随机初始化（这就是"随机权重"的雏形）
model = torch.nn.Linear(1, 1)                    # 输入 1 维 -> 输出 1 维
print(f"初始参数：w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")

# ---- 训练三件套 ----
optimizer = torch.optim.SGD(model.parameters(), lr=2.0)  # lr = 步长
loss_fn = torch.nn.MSELoss()  # 回归任务用均方误差：loss = 平均(pred - y)^2

print("\n开始训练：目标是 w -> 2.0, b -> 1.0\n")

for step in range(100):
    pred = model(x)             # [4,1] 用当前 w、b 算预测值
    loss = loss_fn(pred, y)     # 标量：预测和目标差多远

    optimizer.zero_grad()       # ① 清掉上一步残留的梯度（PyTorch 默认累加）
    loss.backward()             # ② 摸坡：算出 w、b 各自的梯度，存进 .grad —— 参数还没动！
    optimizer.step()            # ③ 迈步：w -= lr * w.grad, b -= lr * b.grad —— 真正更新

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d}  loss = {loss.item():.6f}  "
              f"w = {model.weight.item():.4f}  b = {model.bias.item():.4f}")

print(f"\n学到的规律：y = {model.weight.item():.2f}x + {model.bias.item():.2f}"
      f"（真实：y = 2x + 1）")

# ---- 亲眼看看梯度长什么样（最后一步的）----
print(f"\n最后一步的梯度：w.grad = {model.weight.grad.item():.6f}, "
      f"b.grad = {model.bias.grad.item():.6f}")
print("注意：loss 已经很小，但参数还没有精确到 2 和 1 —— 梯度也变得很小，")
print("这就是 loss 曲线越走越平的原因：坡度本身在变缓。")

# ====== 你的亲手实验（改完先预测再跑）======
#
# 实验 A：把上面 lr=0.1 改成 lr=2.0，先预测 loss 会怎么走，再运行。
#         想想"步子太大"的比喻。
#
# 实验 B：把 optimizer.step() 那行注释掉（行首加 #），先预测会发生什么，再运行。
#         这直接回答 M0 验收题：为什么 backward 之后还要 step？
#
# 实验 C（选做）：lr 改成 0.01（小 10 倍），预测 100 步够不够走到谷底。
