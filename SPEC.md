# Wenyan LLM Lab：从随机权重到后训练与推理优化

版本：1.1 · 2026-10-18 · 用途：交给教学 agent 按里程碑执行。

已确认约束：M4 Pro、48 GiB 统一内存（本机只读系统查询确认）；每周 8–12 小时；云算力预算按人民币理解，优先每月 ≤500 元、硬上限 1,000 元。学习者基础较弱，接受 vibe coding，但希望理解原理。**教学采用 AI 协作实现 + 本人解释/预测/小改动/排错，不要求全项目脱离 AI 手写。**

## 1. 路线决策

采用两段式：**先从零实现并训练一个小语言模型，再在开源预训练模型上做完整的后训练、评测和推理优化。**

| 路线 | 最有价值的收获 | 主要缺口 | 本项目安排 |
|---|---|---|---|
| 直接微调开源模型 | 数据、适配、评测、服务部署 | 容易跳过模型和训练原理 | 第二段主线 |
| 全程只用自训模型 | 权重从哪里来、训练为何成功或失败 | 小数据和算力会限制能力，干扰后训练实验 | 第一段做完整的小闭环 |
| 两段式 | 原理理解与实际效果兼顾 | 需控制范围，避免做两套大工程 | 推荐 |

“从零”定义为：用 PyTorch 张量、autograd、基础层和优化器，自己实现 tokenizer 教学版、Transformer、loss、训练循环与生成循环，权重随机初始化。无需自研自动求导、CUDA kernel 或张量库。

第一段产物准确称为“小型语言模型”，不把它包装成有通用能力的大模型。第二段也不要求选最新或最大的模型。

术语关系：预训练 → 后训练；SFT 本身属于后训练。LoRA/QLoRA 是参数适配方式，DPO 是偏好优化方法，GRPO 是在线强化学习方法。它们不是四个必须依次替换模型的等级。DPO 不等于完成了在线 RL 或完整 PPO-RLHF。[DPO 原论文](https://arxiv.org/abs/2305.18290)、[GRPO 原论文](https://arxiv.org/abs/2402.03300)。

## 2. 学习者背景与项目目标

已查看公开 README 与仓库首页，未进行完整代码审计：

- [facta](https://github.com/Wenyan0315/facta)：有自研 Agent、工具调用、记忆、RAG、OpenAI 兼容网关与评测结构。这为最后的模型接入提供了应用场景。
- [learning-notes](https://github.com/Wenyan0315/learning-notes)：已有 nanoGPT 精读及推理加速相关文章。阅读经历可用于缩短复习，但不能据此认定已掌握实现。
- 学习者目前以 AI coding / vibe coding 为主。核心目标是获得独立解释、修改、诊断和验证模型实验的能力。

完成时，学习者应能：

1. 从一段文本追踪到 token、张量、logits、loss、梯度和权重更新。
2. 独立实现一个可训练的 causal decoder，解释训练和生成的不同。
3. 排查标签错位、未来信息泄漏、mask 错误、过拟合、NaN 和数据污染。
4. 区分预训练、SFT、LoRA、DPO、在线 RL，并各完成对应实验。
5. 用冻结评测证明改善或报告没有改善，而不是只展示好看的样例。
6. 在固定工作负载下测量 KV Cache、量化与推理引擎的效果。
7. 把一个窄任务模型接入 facta，保留质量和性能的对照记录。

## 3. 范围与交付物

必做：小模型预训练、小模型 SFT、小模型 DPO 正确性实验、开源模型 LoRA-SFT、DPO、一次在线 GRPO 实验、自写 KV Cache、量化对比、服务引擎压测、facta 窄场景接入。

后续扩展：QLoRA 对照、完整奖励模型 + PPO、多卡训练、蒸馏、投机解码、自写 Triton/CUDA、MoE。第一轮不以这些为毕业条件。

最终交付：

- 可复现的教学代码库与依赖锁定文件。
- 自训模型权重及模型卡；开源模型 adapter/合并权重的来源与版本记录。
- 数据清单、固定 split、评测集、实验日志和失败案例。
- Base / SFT / DPO / GRPO 分支的质量对比。
- 正确性、显存、吞吐、延迟的推理优化报告。
- facta 接入演示、个人口述答辩和关键模块重写记录。

“跑过”与“学会”分别记录。跑通不自动获得学习验收。

## 4. 时间、硬件与预算

核心里程碑约 **160 小时**，额外预留 20–30 小时用于 Python/PyTorch 基础与重复实验，总学习工作量约 **180–190 小时**。按已确认的每周 8–12 小时，日历安排建议 **20–24 周**，含补基础、排错与机时协调余量。这是规划估算，不是训练时长承诺；以验收门槛推进。

已知本机为 Apple M4 Pro、48 GiB 统一内存；M4 Pro 自带集成 GPU，并非“没有 GPU”。统一内存由系统、CPU 与 GPU 共享，不能视为 48GB 可全部分配给模型。M0 仍需确认系统/框架版本、可用磁盘、月度剩余预算、单次运行时长与成本上限。未知项写 `TBD`。

本项目默认本地写代码、读代码、做数值测试和小模型实验；达到本地时间预算或进入 CUDA 专项时再租 Linux 单卡环境。云端优先比较 24GB 显存档的实际报价和软件兼容性，不为教学默认租多卡或高价旗舰卡。

### 本人的预算执行方案

- 月度正常计划上限 500 元，硬上限 1,000 元；500–1,000 元是已有预算中的机动空间，不是默认消费目标。每次拟超正常计划先呈现原因和剩余额度。
- 建议 500 元内部预留：计算 350 元、存储/传输等附加费用 50 元、失败重跑 100 元。实际费用项目以云平台为准。
- 先做一场预计不超过 30 元、最长约 2 小时的云端环境验证；这是建议的首次实验限制，不是 GPU 市场报价。
- 只按需开实例。写代码、读资料和等 agent 回复尽量在本地完成；停掉训练进程不一定等于停止云实例计费。
- AutoDL 可作为国内按时租用的候选。其官方说明区分开关机计费与付费数据盘计费，选择前检查实时报价、存储费用和数据保留规则。[官方计费说明](https://www.autodl.com/docs/price/)。本计划不声称已经取得某型号报价。
- 预算例子仅用于算术：若实际单价是 5 元/小时，则 350 元计算预算约 70 小时；若是 10 元/小时，则约 35 小时。两者都不是平台报价或训练完成保证。
- 单次任务保存 checkpoint、设置硬性步数/时长上限，完成后备份产物并确认实例停止；不能仅靠聊天中的“训练完成”判断已停止计费。
- 当前请求只制定学习计划，**本次没有购买、租用或启动任何收费实例**。

### 建议日历

| 周次 | 重点 | 主要运行位置 |
|---|---|---|
| 1–2 | 基础过渡 + M0：Python、张量、autograd、最小训练 | Mac |
| 3–4 | M1–M2：tokenizer、attention、decoder | Mac |
| 5–6 | M3：预训练与故障诊断 | Mac 起步，按实测决定云端正式跑 |
| 7–9 | M4–M5：结构、SFT、LoRA/DPO 最小实现 | Mac 为主 |
| 10–12 | M6：开源模型数据与 LoRA-SFT | Mac 准备，云端集中训练 |
| 13–15 | M7–M8：DPO 与在线 GRPO | 云端短实验 + Mac 分析 |
| 16–18 | M9：KV Cache 与数值、性能验证 | Mac |
| 19–21 | M10：量化与服务引擎压测 | Mac 后端 + 按需云端 vLLM |
| 22 | M11：facta 接入与答辩 | Mac + 选定服务 |
| 23–24 | 补实验、整理作品与回顾 | 按需 |

能直接通过的部分可以压缩，目标区间为 20–24 周，不以日历日期强行判过。

| 可用条件 | 第一段建议 | 第二段建议 |
|---|---|---|
| CPU 或较小内存设备 | 约 1–5M 参数，短上下文，优先正确性 | 先运行小模型基线；训练/引擎实验按实测决定是否需要云 GPU |
| Apple Silicon | PyTorch CPU/MPS 完成教学实验，先查算子与精度支持 | 本地兼容推理后端；CUDA 训练与 vLLM 实验可集中到 Linux GPU 环境 |
| NVIDIA 12–16GB | 约 10–30M 参数教学模型 | 0.6B 级 LoRA 起步；DPO、GRPO 分别做显存探测 |
| NVIDIA 24GB 及以上 | 同样先做小模型，不因显存大而扩范围 | 0.6B 仍为默认，可在明确需要时升到约 1–2B |

以上是起步建议，不保证给定 batch、序列长度、精度或框架版本一定能装下。Apple 统一内存不能直接等同 CUDA 显存；不要默认某个 CUDA 量化方案在 MPS 可用。

每次正式训练前：

1. 跑 20–100 步代表性 smoke test，记录有效 tokens/s、峰值内存、保存和验证耗时。
2. 用 `预计训练秒数 ≈ 计划处理 token 数 / 实测有效 tokens/s`，再加评估、保存和启动余量。
3. 云成本用实时报价 × 预计占用时间计算，并设置最大步数、最大时长、停止条件。
4. 先降低 microbatch、上下文或启用适用的内存优化，再考虑升级硬件。
5. GRPO 额外测 rollout 时间、采样条数和生成长度，不能沿用纯 SFT 的吞吐估算。

教学 agent 不得因本 SPEC 自动购买算力或运行无预算上限的任务；用户已明确授权的预算范围内无需重复询问。

## 5. 两段模型与一个应用任务

### A. Tiny-Wenyan：用于理解机制

首版先实现最简单的 GPT 风格 decoder：causal MHA、位置编码、LayerNorm、MLP、残差。正确后逐项实验 RoPE、RMSNorm、SwiGLU；不要一次换完再猜收益。

目标配置可取 6 层、hidden 384、6 个注意力头、head_dim 64、词表约 8K、上下文 256，使用 tied embedding；换成 SwiGLU 时中间维度可取 1024，总参数约 14M，以代码统计为准。资源不足时降至约 1–5M。

字符/字节 tokenizer 用于最初调试；学习者另写小语料上的 BPE merge 教学实现。正式小模型训练可使用成熟 tokenizer 库训练并冻结一个 BPE，避免把大量时间花在 tokenizer 性能工程上。

默认预训练语料候选为 [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 的固定子集。它是英文教学数据：这一段验证语言建模，不以中文助手能力验收。数据使用前记录许可证、版本和选择原因。若换中文语料，重新评估词表、数据质量与 token 预算，不默认效果相当。

先用约 1–5M tokens 跑通链路，再按测量结果安排约 20–100M tokens 的学习实验。这是预算档位，可能明显训练不足，不是优质通用模型的配方。不得通过反复重复少量语料来冒充独立数据量。

### B. Open-Wenyan：用于实际后训练

默认候选是 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，其官方模型页提供权重、许可证和使用信息。选择原因是规模小、可用预训练权重和适合做中文窄任务实验；不是“最新模型”承诺。开始时验证可下载性与选定软件版本的兼容性，锁定 model revision、tokenizer revision。

从 Base 做 SFT 才能观察本项目的指令适配过程。可额外用同系列官方后训练模型作为参考，但明确其已经接受过后训练。Base 的聊天模板必须检查实际文件；若没有合适模板，项目定义并冻结自己的模板，训练、评测和服务全程一致。

分支关系：

```text
随机权重 → Tiny 预训练 → Tiny SFT → Tiny DPO 正确性实验
                       └→ 手写生成与 KV Cache

开源 Base → LoRA-SFT ┬→ DPO：离线偏好优化
                    ├→ GRPO：在线采样 + 可验证奖励
                    └→ SFT 对照

通过质量筛选的 checkpoint → 推理优化 → facta 窄任务接入
```

DPO 与 GRPO 默认从同一个 SFT checkpoint 分叉，便于观察不同目标的作用；不要求 DPO 后必须继续 GRPO。

### C. Facta-Micro：贯穿第二段的窄任务

建议任务：**中文笔记查询场景的决策与结构化输出**。输入包含用户请求、必要上下文和固定协议；输出：

```json
{"route":"search_notes","query":"上周讨论过的缓存优化","response":""}
```

`route` 只能是 `direct`、`search_notes`、`clarify`。`search_notes` 要求 query 非空、response 为空；另两种要求 query 为空、response 非空。`clarify` 的 response 是澄清问题，`direct` 的 response 是直接回答。

示例类别：用户要求查自己的笔记 → 检索；缺少判断所必需的信息 → 澄清；无需个人知识的简单问题 → 直接回答。先写明确标注规则，再标注数据。

这是一个项目自定义任务协议，**不宣称与 facta 现有路由字段或工具接口完全相同**。接入阶段阅读代码后写 adapter，并验证映射。训练阶段用 mock 笔记和工具，不依赖真实终端操作。

模型只承担这个窄任务。通用多步 Agent、复杂工具调用和长程推理不作为 0.6B 教学实验的效果承诺。

## 6. 里程碑与验收

工时为学习者动手时间的建议分配，共约 160 小时，不含所有机器等待时间。

**弱基础适配规则优先于下方“独立完成”的字面要求：** 下方列出的模块是本人必须理解和参与的部分，不要求第一次就白纸手写。可先让 agent 给出完整、最小、可运行实现，然后本人追踪一个样本、完成一个 5–30 行的局部修改或补全、预测并验证一次行为变化。最终只对关键小片段做脱离参考答案的挑战，不考记忆 API。本人需要解释并实际修改；不能只运行 agent 的脚本。

### 基础过渡：额外 20–30 小时，贯穿前几周

只补项目会用到的内容：Python 函数/类与循环、list/dict、读报错与 traceback；张量 shape、索引、广播、矩阵乘法；概率与 log、导数和链式法则直觉；PyTorch autograd、训练/验证分离。

不用先学完线性代数或高等数学。每次引入一个概念，配一个小数值例子和不超过几十行的程序。依次完成：修改一个 Python 函数 → 解释广播 → 用 autograd 拟合直线 → 训练小分类器 → 预测下一 token。卡在数学符号时先画数据流，再回到公式。

通过条件：能读懂并修改小程序、说明每个主要 tensor 的 shape、解释 loss 降低为何不保证新数据表现更好。达不到则缩小题目，不用增加长篇讲义或更多框架。

### M0：诊断与实验环境（6 小时）

任务：检查 Python/PyTorch 基础，确认硬件预算，初始化最小项目、锁依赖、记录 seed 与环境。先跑一次开源模型原始输出，获得直观基线。

学习者独立完成：解释 tensor 的 shape、矩阵乘法、softmax、交叉熵、梯度下降；写一个线性分类器的 forward/backward/update。不会的部分按需补，不要求先补完整套数学课程。

验收：能解释为什么调用 backward 后还要 optimizer.step；能画出 `[B,T] → [B,T,V]`；记录一个真实环境 smoke test。不能用“读过笔记”替代。

交付：`docs/baseline.md`、`configs/environment.yaml`、能力诊断结果。

### M1：tokenizer、数据与 next-token 目标（8 小时）

任务：字节/字符编码 → 小型 BPE → 成熟库 tokenizer；构造 shifted labels、padding、EOS 与 attention mask。展示一条样本经过所有转换的结果。

学习者独立完成：BPE 合并核心循环、batch 切片、next-token 对齐的小例子。允许慢，不要求工业 tokenizer 性能。

验收：encode/decode 往返规则清楚；理解有损规范化的例外；能指出哪个 logit 预测哪个 token；数据先按文档/重复组划分，再切 chunk。训练 tokenizer 不使用验证和测试文本。

实验：同一文本用两种词表编码，比较序列长度；不直接跨 tokenizer 比较 perplexity 高低。

交付：`src/data/`、`data/manifest.json`、一页数据流解释。

### M2：手写 decoder-only Transformer（14 小时）

任务：先做单头 attention，再多头，再 decoder block 和整模型。标注 Q/K/V、attention scores、残差与 logits 的 shape。

学习者独立完成：scaled dot-product attention、causal mask、一个 decoder block。先不调用完整 Transformer 模型封装。

验收：改变未来 token 不影响之前位置的输出；一个小 batch 的梯度存在且有限；在 eval、相同权重与 mask 下，对照框架参考 attention 的数值结果，记录 dtype 对应容差。

实验：移除 causal mask 展示信息泄漏；恢复正确版本后再进入正式训练。

交付：`src/tiny_model/`、参数量报告、因果性与数值测试。

### M3：真正从随机权重预训练（18 小时）

任务：实现训练循环、AdamW、学习率调度、梯度裁剪、适用的混合精度、梯度累积、保存恢复、验证和采样。

学习者独立完成：训练循环的 loss/backward/update 核心，解释梯度累积的缩放与有效 batch。记录 `有效 batch tokens = microbatch × context × accumulation × world_size`，存在 padding 时另报非 padding tokens。

验收分三步：

1. 极小固定 batch 可以过拟合，例如 CE 降到 0.1 以下；若数据存在不可消除的目标冲突则单独说明。
2. 独立验证集 loss 相对随机初始化基线下降；训练与验证曲线、固定 prompt 的阶段采样同时保存。
3. 中断恢复包含模型、优化器、scheduler、步数、随机状态与数据位置；在允许的数值误差内验证续训一致性，不能只验证权重能加载。

实验：学习率过大、训练样本过少各一次，先预测现象再观察。达到预算上限后报告能力限制，不无限加训练。

交付：至少一个预训练 checkpoint、曲线、数据/token 使用量、失败日志。

### M4：现代结构与小模型 SFT（14 小时）

任务：用独立分支逐项学习 RoPE、RMSNorm、SwiGLU；至少完成一次结构对照。再对一个选定架构的已训练 checkpoint 做 SFT。

重要：结构改变后不能把旧结构权重当成完全兼容；需要明确转换或重新训练。结构对照固定数据与训练 token 预算，并报告参数量差异。

Tiny SFT 用与预训练语言相符的简单任务：按指定格式续写、短文本提取或简单问答，约 200–1,000 条教学样本起步。中文 Facta-Micro 留给第二段。

学习者独立完成：response-only loss mask，打印实际参与 loss 的 token，解释 prompt、padding 和 EOS 的处理。

验收：预训练模型与 SFT 模型使用同一冻结任务集比较；能区分格式改善和内容正确；保留预训练验证集观察遗忘。不得仅看 SFT training loss。

交付：结构消融记录、小模型 SFT checkpoint、mask 测试与前后对照。

### M5：LoRA 与 DPO 的最小实现（10 小时）

任务：手写一个 LoRA Linear；手写序列 log-prob 和 DPO loss。先用人工张量验证，再在 Tiny SFT 上做短训练，约 100–300 个偏好对即可起步。

学习者独立完成：`W_eff = W + (alpha/r)BA` 的维度说明、冻结基础权重、adapter 合并；DPO 的 chosen/rejected log-prob 差与 reference 的关系。

验收：LoRA 初始化为 no-op 时输出一致，训练后基础参数未变化，合并前后输出相符；DPO 的 prompt/padding 被排除，reference 冻结，chosen/rejected 交换改变优化方向。记录 sequence sum 与 length normalization 的选择，不能默默混用。

Tiny DPO 只验机制、梯度和行为变化，不要求通用对齐能力。公式和实现对照 [DPO 原论文](https://arxiv.org/abs/2305.18290)，LoRA 工程细节参考 [PEFT 官方文档](https://huggingface.co/docs/peft/package_reference/lora)。

交付：可读的最小实现、数值验证、小模型偏好实验报告。

### M6：开源 Base 模型的 LoRA-SFT（16 小时）

任务：冻结 Facta-Micro 协议和评测，建立 Base 基线，再用 Transformers + PEFT 或 TRL 完成 LoRA-SFT。第一轮先 LoRA，只有内存需求或明确学习目标时再做 QLoRA。

数据起步：约 1,000–5,000 条训练、200 条验证、200 条冻结测试；是学习规模，不是效果保证。按意图家族/生成模板/来源分组隔离，避免只把同一句改写随机分到两边。

学习者独立完成：人工检查 30 条样本、写一个 data collator 或解释框架 collator 的实际张量、配置 LoRA targets 并核对可训练参数。

验收：Base / 提示词改进基线 / SFT 使用相同协议；报告路由正确率、JSON/schema 合格率、查询约束通过率及错误类型。保存 adapter、tokenizer、模板、base revision，重载后复现输出。

至少一次单变量对照：数据清洗前后或 rank 改变。训练预算有限时只选一个，不扫大网格。SFT mask 行为按锁定版本验证，参考 [TRL SFT 文档](https://huggingface.co/docs/trl/sft_trainer)。

交付：数据卡、SFT adapter、评测结果与模型卡。

### M7：开源模型 DPO（16 小时）

任务：从同一 SFT checkpoint 采样响应，制作约 500–2,000 组偏好数据，验证约 100–200 组。优先使用实际候选错误，避免所有 rejected 都只是明显乱码。

偏好标准：任务正确优先于格式与简洁；不把越长越好或越短越好当作正确性。每个偏好对记录来源和判断依据，学习者审查至少 30 对；合成/模型标注不称为真人偏好。

学习者独立完成：手算一个偏好对的 DPO loss，说明 beta、reference、chosen 和 rejected 的角色。

验收：reference 必须是本阶段起点 SFT 的冻结分布，不能误用关闭 SFT adapter 后的原始 Base。采用显式 reference、缓存 reference log-prob 或其他方案时都要验证它代表哪个模型。

报告 SFT vs DPO：任务指标、响应长度、偏好胜率与退化样例；偏好 margin 变大不能替代任务质量。只用验证集选择超参，冻结测试只用于阶段报告。[TRL DPO 文档](https://huggingface.co/docs/trl/dpo_trainer)。

交付：偏好数据版本、DPO 模型、至少一次单变量实验和负面案例。

### M8：在线 GRPO 小实验（14 小时）

任务：从 M6 的 SFT checkpoint 另开分支，执行 rollout → reward → advantage → update → evaluation。使用 Facta-Micro 的可程序验证子集；如候选奖励全为零，改用更简单的路由任务或单独的短算术任务，并注明任务变更。

先在几十个 prompt 上探测模型是否会产出有对有错的候选；每个 prompt 可从 4 个候选起步，视内存缩小，至少保留可用的组内比较。全组同奖时没有有效的相对信号，应诊断任务难度、采样与奖励。

学习者独立完成：奖励函数、一个组的 advantage 手算、至少 5 个针对奖励漏洞的反例。生成内容只解析为数据，不用 eval 执行。

奖励优先反映路由/答案正确；格式分不能压过语义正确。训练 verifier 与离线评测器分别审查，额外使用未参与奖励设计的人工样例。记录 reward、离线准确率、长度、组内奖励方差及实际配置启用的 KL 项。

验收：发生真实在线采样和参数更新，记录所用 GRPO 变体、loss normalization、clip、KL/beta 与 reference 设置。不能将离线 DPO 改名 RL，也不能只运行奖励函数就称完成 GRPO。出现 reward 上升而独立正确率下降时，应判为 reward hacking 并分析。[TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)。

结果可为无增益：必须有实际实验、诊断和下一步假设。算力不足时标记 `deferred`，不能把必做项静默改为完成。完整 RM + PPO 是第二轮扩展。

交付：奖励实现、在线训练记录、SFT vs GRPO 对照和漏洞报告。

### M9：手写推理与 KV Cache（16 小时）

任务：在 Tiny 模型实现 temperature、top-k/top-p、EOS 停止、prefill 和逐 token decode，再加入每层 KV Cache。先单序列，再做简单 batch/padding。

学习者独立完成：cache 更新、位置索引和 mask，解释为什么保存 K/V、为什么 prefill 和 decode 的工作不同。

验收：eval 模式下固定前缀和相同 token 序列，比较有无 cache 每步 logits；验证初始 prefill、后续单 token、padding、上下文上限与 cache 重置。贪心生成在无数值临界差异时应一致；不能只看随机生成文本是否相似。

实验：短/长 prompt 与不同输出长度的耗时；说明 cache 可能以更多显存换速度。估算 `KV bytes ≈ 2 × layers × batch × cached_tokens × kv_heads × head_dim × bytes_per_element`，并说明实际后端额外开销。

交付：KV Cache 实现、正确性测试、prefill/decode 分项测量。

### M10：量化、服务引擎与公平压测（20 小时）

任务：选一个通过质量评估的开源 checkpoint，对照普通框架生成与成熟推理引擎；完成一次真实权重量化及质量回归。

CUDA 环境优先评估 [vLLM](https://github.com/vllm-project/vllm)；CPU/Apple 环境可评估 [llama.cpp](https://github.com/ggml-org/llama.cpp)。安装时核验模型、量化格式、硬件和版本支持，不把自定义 Tiny 架构导入 vLLM 作为前置条件。

顺序：原始精度基线 → 同精度引擎对照 → 量化对照 → 并发对照。每次尽量只改一个变量。先合并/导出并验证模型身份，再量化；QLoRA 训练格式不自动等于部署格式。FlashAttention/SDPA 可以做附加实验，但不能把调用 API 等同于已确认使用某个 kernel。

工作负载至少覆盖 prompt 长度 128/512、输出上限 64/128、并发 1/4；资源允许再增至 16。容量不足的格子如实记 OOM。固定模型、tokenizer、模板、输入、生成参数、硬件与请求集合。

每组预热，重复至少 3 轮；单请求计时同步加速器。分别报告 TTFT、TPOT（输出超过 1 token 才有定义）、端到端延迟 p50/p95、整体输出 tokens/s、峰值内存、错误率、实际输出长度。短跑 p95 不稳定时披露样本量。区分冷启动、稳态、引擎内部和 HTTP 端到端耗时。

自然 EOS 的真实任务测试与固定输出长度的性能测试分开；后者若强制长度必须标注。关闭或明确控制 prefix cache，防止重复请求缓存造成虚假提速。

学习者独立完成：一段基准计时代码，并解释低并发延迟与高并发吞吐为何可能得出不同结论。

验收：至少证明一项测得的收益或用 profile 解释无收益；量化前后跑同一质量集。建议预先设定“主要任务指标下降不超过 2 个百分点”的候选阈值，结合样本量和业务需求解释，不将小样本差异当作确定结论。超阈值则保留较高精度版本。

交付：原始压测记录、质量/延迟/吞吐/内存表、可复现配置和取舍结论。

### M11：接回 facta 与毕业验收（8 小时）

任务：阅读 facta 当前模型网关和路由实现，给 Facta-Micro 写最小适配层。在隔离配置中接到兼容服务，使用冻结场景对照原模型与新模型；关闭或单独记录自动 fallback，不能把回退模型的答案记为学生模型效果。

接入验收：至少 20 个固定端到端场景，涵盖正常查询、澄清、格式错误、超时与回退。分别报告模型决策、工具执行和任务最终成功，避免把检索质量变化归因于训练。

学习验收：在给定接口/骨架下，不看参考答案补全一个约 10–30 行的关键片段，例如 label shift、loss mask 或 KV 更新；诊断一个未见过的局部错误，并完成约 20 分钟口述讲解。允许查 API，不要求背诵整段模型实现。

交付：演示、最终技术报告、个人复盘、未完成项和下一轮学习建议。

## 7. 评测与数据规则

1. **先评测后训练**：M1 固定语言建模 split；M6 固定应用任务 split。训练、验证、测试各有用途。
2. **先分组再划分**：近重复文本、同源模板、同一笔记/问题家族尽可能留在同一 split，记录去重规则与哈希。
3. **不可拿测试集调参**：测试暴露后需迭代时，保留旧集作为回归集，另外建立未见测试集；不得悄悄重标难题。
4. **实际正确性优先**：合法 JSON 不等于路由正确；偏好胜率不等于知识准确；训练 reward 不等于泛化。
5. **可追溯合成数据**：保留生成来源、模型/提示词版本、人工检查结果。只将已授权或可用的数据纳入训练；个人笔记默认使用脱敏或 mock 数据。
6. **统计诚实**：报样本量、成功数、失败数和百分点。核心对比条件允许时做配对 bootstrap 或第二个 seed；单次小样本结果注明不确定性。
7. **记录退化**：每阶段至少保存 5 个失败案例；若不足则报告全部。未提升可以构成有效学习实验，但不能宣称模型改善。
8. **有意义的测试**：优先测试因果性、loss mask、reference、缓存等容易隐性出错的行为，不为凑覆盖率写机械测试。

## 8. 建议工程结构

以下是未来学习仓库的设计，不代表本次已经实现这些文件。

```text
wenyan-llm-lab/
  AGENTS.md
  SPEC.md
  pyproject.toml
  configs/
  src/
    data/
    tiny_model/
    training/
    post_training/
    inference/
    evals/
  scripts/
  tests/
  data/manifest.json
  experiments/          # 配置、指标、样例、实验结论
  docs/learning/        # 本人写的解释、预测、复盘
  docs/model_cards/
  progress/STATUS.md
  progress/HANDOFF.md
  checkpoints/          # 权重不直接进入普通 Git 历史
```

建议命令契约：`prepare-data`、`train-tiny`、`sft`、`dpo`、`grpo`、`evaluate`、`benchmark`、`serve`。由教学 agent 逐阶段实现，不要第一天搭建完整平台。

每次实验至少记录：run_id、git commit、环境锁文件、硬件、模型与 tokenizer revision、数据版本、seed、超参、实际 tokens、耗时、成本（若有）、峰值内存、指标、固定采样、失败原因。所有表格中的数字必须来自实际记录；示意数据需显式标注。

## 9. 教学协议：让 AI 带学，而非代学

每节 60–90 分钟建议节奏：5 分钟回忆 → 10–15 分钟用一个例子解释 → 10 分钟阅读 agent 给出的最小实现并标 shape → 20–30 分钟本人修改/预测 → 15–20 分钟实验与排错 → 5–10 分钟复述和记录。超时就拆课，不一次堆多个新概念。

agent 可以负责环境、下载、脚手架、绘图、实验记录、代码审查，也可以生成完整的模型与训练第一版。代码优先短、直白、少抽象。学习者必须参与 attention、label shift、loss mask、训练 update、LoRA、DPO log-prob、reward 与 KV Cache 的关键实现或受控修改。

默认采用示范 → 引导练习 → 变式小挑战。可以先给完整最小代码和行内解释，再让学习者选择或预测一个变化，并实际改动验证。逐步减少提示，不以拒绝提供代码来强迫学习。卡住两个短回合就缩小问题或给局部参考解，随后安排另一个变式。将“看过答案”“提示后理解”“独立通过”分开。

每个原理至少过四关：用自己的话解释它解决什么问题；追踪一个具体输入的 shape/数值；预测改动的后果；用实验确认或修正预测。背定义和复述 agent 的文字不作为唯一证据。

每个里程碑从四个维度评分，每项 0–2 分：

- 解释：能否说清输入、输出、目的与假设。
- 实现：能否独立写出或改变关键逻辑。
- 排错：能否通过证据定位一个错误。
- 证据：能否设计公平对照并解释结果。

通过建议：总分至少 6/8，且“实现”“排错”均不为 0。已有经验者可用直接挑战题免修，但需保存证据。本人没有作答时 agent 不得替其生成答案并判通过。

## 10. Agent 分工和交接

一个主导师足够；用户若使用多个 agent，可分为导师、实现助教、实验审计员、答辩考官。角色可由同一个 agent 分时承担。禁止多个 agent 同时改同一核心模块。

主导师负责当前阶段和教学节奏；助教维护非教学脚手架；审计员检查数据泄漏、实验公平性和成本；考官出变式题并核实学习证据。

每次结束更新：当前里程碑、已完成证据、学习者实际能解释的内容、尚不理解的内容、下一项 30–90 分钟任务、运行中的实验、预算消耗。`done`、`assisted`、`independent`、`deferred` 分开记录。

详细可复制角色提示词见配套 `AGENT-HANDOFF.md`；初始化状态模板见 `LEARNING-STATUS.md`。

## 11. 资料使用顺序

每阶段只读解决当前问题的部分，避免收集一堆课程却不实验。

| 阶段 | 主参考 | 使用方式 |
|---|---|---|
| M0–M4 | [Stanford CS336 2025](https://cs336.stanford.edu/spring2025/) | 按基础建模、数据与训练问题选读，不先完整刷完 |
| M1–M3 | [nanoGPT](https://github.com/karpathy/nanoGPT) | 自己尝试后对照 model/train 结构，不直接复制当作从零实现 |
| 全流程补充 | [MiniMind](https://github.com/jingyaogong/minimind) | 看训练阶段如何串联；其成本与效果不能直接套到本项目 |
| M5–M6 | [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)、[TRL SFT](https://huggingface.co/docs/trl/sft_trainer) | 最小实现之后使用工程库，核验版本行为 |
| M7 | [DPO 论文](https://arxiv.org/abs/2305.18290)、[TRL DPO](https://huggingface.co/docs/trl/dpo_trainer) | 对齐公式、reference 与数据协议 |
| M8 | [DeepSeekMath](https://arxiv.org/abs/2402.03300)、[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) | 理解组相对信号，记录实现变体 |
| M10 | [vLLM](https://github.com/vllm-project/vllm)、[llama.cpp](https://github.com/ggml-org/llama.cpp) | 查兼容性和测量方法，再实测本机效果 |

资料核验日期为 2026-10-18。安装 API、支持硬件和量化格式以实际采用版本的官方文档为准；实验环境稳定后锁版本，避免学习过程无目的追新。

## 12. 项目完成定义

- [ ] Tiny 模型确实由随机权重训练，且正确性测试和训练记录完整。
- [ ] Tiny SFT/DPO 与开源 SFT/DPO 都有实验结果。
- [ ] 至少一次真实在线 GRPO 实验完成；未做则明确项目尚有该项未完成。
- [ ] 手写 KV Cache 通过数值检查；成熟引擎与量化有公平对照。
- [ ] Facta-Micro 接入有固定场景回归，不把失败回退隐藏起来。
- [ ] 学习者通过关键模块挑战题与答辩。
- [ ] 最终报告包括负面结果、成本、限制、可复现步骤和下一步。

能力没有显著提升时，可以完成一次严谨的学习项目；但只有脚本运行成功、缺乏本人理解和评测证据时，不能算完成本 SPEC。
