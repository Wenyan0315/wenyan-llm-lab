# M0 Baseline：环境与能力诊断记录

日期：2026-10-18 · 状态：M0 完成

## 环境快照（真实查询结果）

| 项目 | 值 |
|---|---|
| 硬件 | Apple M4 Pro（Mac16,7），48 GiB 统一内存（51,539,607,552 bytes） |
| 系统 | macOS，磁盘 460Gi 总量 / 349Gi 可用 |
| Python | 3.14.7（venv：`.venv/`） |
| PyTorch | 2.14.1（`pip freeze` 已锁入 `requirements.txt`） |
| numpy | 2.5.4 |
| MPS | `torch.backends.mps.is_available() == True` |
| 云资源 | 未租用任何实例，本月本项目花费 0 元 |

重建环境：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Smoke Test

- Lesson 02 拟合直线：100 步 SGD，w: -0.0075 → 1.9996，b: 0.5364 → 1.0008，loss 17.1 → 5e-7 ✅
- Lesson 03 线性分类器：100 样本 2 分类，acc 0.61 → 0.94，未见过测试点分类正确 ✅

## M0 验收对照

| 验收项 | 结果 | 证据 |
|---|---|---|
| 解释为什么 backward 后还要 step | ✅ | 独立答出"梯度算了但没生效"；实验 B（注释 step）loss 全程恒定 |
| 画出/推导 [B,T] → [B,T,V] | ✅ | Lesson 03 Q1：[100,2]@[2,2]+[2] → [100,2]，独立答对消耗维与广播 |
| 真实环境 smoke test | ✅ | 上表两项 + MPS 验证 |

## 能力诊断结果（2026-10-18 初始诊断 + 三课记录）

| 概念 | 状态 | 备注 |
|---|---|---|
| Python 切片/基础 | independent | 切片题直接答对 |
| tensor shape / 广播 / matmul | assisted | 预测全对；"广播=复制而非消耗"经纠正掌握 |
| softmax / 交叉熵 | assisted 偏 independent | 稀释机制独立答对；数值锚点（loss 地板 0、ln2≈0.69）已建立 |
| 训练循环三步舞 | assisted | Q1 独立答对；lr 发散预测命中（17→1.6e25→inf→nan） |
| label shift | seen | 切片会，机制已讲解，M1 变式再验 |

## 未完事项

- 过拟合实测（"loss 降低为何不保证新数据更好"）→ 留在 M3 过拟合实验正式验证
- 开源模型原始输出基线（原 M0 项）→ 推迟到 M6 前的 Base 基线（避免现在下载 0.6B 模型）
