# Wenyan LLM Lab：学习状态

更新时间：2026-10-18（Lesson 02 结束后）

## 资源与环境（M0 已确认）

| 项目 | 状态 |
|---|---|
| Python | 3.14.7（系统框架版） |
| PyTorch | 2.14.1（wenyan-llm-lab/.venv，MPS available: True） |
| 依赖 | 尚未锁版本；numpy 未装（torch 有无害 warning），M0 收尾时补装并生成 requirements |
| 磁盘 | 460Gi 总量，349Gi 可用 |
| 内存 | 51,539,607,552 bytes ≈ 48 GiB |
| 云预算 | 未租用任何实例，本月本项目花费 0 元 |

## 当前状态

- 当前里程碑：基础过渡 / M0，**in_progress**
- 已完成：环境诊断、PyTorch/MPS smoke test、Lesson 01（tensor shape / 广播 / softmax / 交叉熵 / label shift 入门）、Lesson 02（autograd / 训练三步舞 / lr / 梯度累加）
- 下一课：Lesson 03 — 训练小分类器（M0 学习者独立完成项：线性分类器 forward/backward/update 全链路）
- M0 剩余：线性分类器、依赖锁定（numpy + requirements）、docs/baseline.md
- 运行中的任务：无

## 关键概念掌握表

| 概念 | 状态 | 证据 | 下次变式题 |
|---|---|---|---|
| tensor shape 与广播 | assisted | Lesson 01 预测 z=[2,3,5]、m=[2,3,8] 全对 | 不看讲义解释 5 为何消失、3 为何保留 |
| next-token 与 label shift | seen | 切片 [:-1]/[1:] 独立答对；错位原理由 agent 讲解 | 给 4 个 token 写出 x/y 并解释为何错位 |
| softmax 与交叉熵 | assisted 偏 independent | Q2（稀释机制）独立答对；Q1 方向对、数值锚点（loss 地板=0）刚补 | 换 target 后预测 loss 变化 |
| 梯度与参数更新 | assisted | Q1 独立答对（无 step 则参数不动、loss 恒定）；Q2 预测发散命中（lr=2.0 实测 17→1.6e25→inf→nan）；Q3 梯度累加经对照实验讲解 | 用自己的话解释 zero_grad 位置与梯度累积的关系 |
| attention 与 causal mask | unverified | — | — |
| SFT loss mask | unverified | — | — |
| LoRA / DPO / GRPO / KV Cache / 量化 | unverified | — | — |

## 里程碑看板

| 阶段 | 执行状态 | 理解状态 |
|---|---|---|
| 基础过渡 | in_progress（Lesson 01 done） | assisted |
| M0 诊断与环境 | in_progress（环境 done，诊断题 done，线性分类器未做） | assisted |
| M1–M11 | not_started | unverified |

## 预算记录

| 月份 | 项目实验已用 | 剩余正常计划 | 硬上限剩余 |
|---|---|---|---|
| 2026-10 | 0 元 | 500 元 | 1,000 元 |

## 课程记录

### Lesson 01（2026-10-18）：shape / 广播 / softmax / 交叉熵

- 诊断：Python 切片 ✓；shape/广播 ✗；softmax/CE ✗ → 本课补
- 亲手实验：my_logits [2.0, 0.0, 3.5] 使 loss 3.17 → 0.226，方向正确
- 变式：Q1（正确类别 logit→10，loss→0.0004）方向对、数值预期错（以为接近 3）；Q2（非正确类别→10，loss→6.5）独立答对
- 关键锚点已建立：loss = -ln(p[正确])，地板 0；logit 大 → e 的幂暴涨 → 抢蛋糕
- 脚本：lessons/lesson01_tensor_loss.py

### Lesson 02（2026-10-18）：autograd 拟合直线

- 内容：训练三步舞 zero_grad/backward/step、lr=步长、loss 前陡后平（坡度变缓）、"出题器/数据/模型猜测"三层区分、泛化雏形（预测未见过的 x=5/10/100 全对）
- 亲手实验：A）lr=2.0 → 一步 w 跳到 30.9，10 步 loss 1.6e25，20 步 inf，40 步 nan；C）无 zero_grad → 梯度累加，loss 在 1~21 乱跳不收敛（step 0/1 与对照组相同，step 2 起分岔）
- 解释题：Q1 独立答对；Q2 方向命中但正反馈机制需补全；Q3 未答出（梯度默认累加），经对照实验讲解
- 诚实记录：基准版在预测前已被本人运行过；预测 3 仍错（把基准"下降趋平"错搬到无 step 场景）——"学习只发生在 step"由此确立，预测证据按 assisted 记
- 伏笔埋下：nan 排查（M3）、梯度累积=设计化的累加（M3）、过拟合=硬背不泛化（M3/M4）
- 脚本：lessons/lesson02_autograd.py
