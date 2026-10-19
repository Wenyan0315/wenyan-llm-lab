# Wenyan LLM Lab：学习状态

更新时间：2026-10-19（Lesson 08 结束，M1 概念部分完成）

## 资源与环境（M0 已确认）

| 项目 | 状态 |
|---|---|
| Python | 3.14.7（系统框架版） |
| PyTorch | 2.14.1（wenyan-llm-lab/.venv，MPS available: True） |
| 依赖 | 已锁定：requirements.txt（torch 2.14.1、numpy 2.5.4 及传递依赖） |
| 磁盘 | 460Gi 总量，349Gi 可用 |
| 内存 | 51,539,607,552 bytes ≈ 48 GiB |
| 云预算 | 未租用任何实例，本月本项目花费 0 元 |

## 当前状态

- 当前里程碑：M0 **done**；M1 **概念部分完成**（Lesson 04–08），剩交付物（真实语料 + 成熟库 tokenizer + src/data/ + data/manifest.json）
- 已完成：Lesson 01（shape/广播/softmax/CE）、Lesson 02（autograd/三步舞/lr/梯度累加）、Lesson 03（从零线性分类器 + 数据可分性）、M0 交付、Lesson 04–06（字符级 / byte-level / BPE）、Lesson 07（真实文本 shifted labels + chunk）、Lesson 08（split 规则与泄漏实验）
- 下一课：M1 交付物——TinyStories 子集（记录许可证/版本/选择原因）、成熟库 tokenizer 只在 train 文本上训、src/data/ 与 data/manifest.json
- 运行中的任务：无

## 关键概念掌握表

| 概念 | 状态 | 证据 | 下次变式题 |
|---|---|---|---|
| tensor shape 与广播 | assisted | Lesson 01 预测 z=[2,3,5]、m=[2,3,8] 全对 | 不看讲义解释 5 为何消失、3 为何保留 |
| next-token 与 label shift | assisted | Lesson 01 切片 [:-1]/[1:] 答对；Lesson 07 用「输入没尾巴、预测没头」独立说清 x/y 各少一个；V=词表大小、块硬切不看词边界由 agent 讲解 | 给一条 5 token 句子写 x/y 并说出 logits 的 V 是多少 |
| tokenizer（字符 / 字节 / BPE） | assisted 偏 independent | Lesson 04 暴露"按词切"直觉（22 字符猜成 6）；Lesson 06 预测 2/3 命中（第 1 轮 lo、lowkey 不崩）；Lesson 07 块吃 13 出 12 已理解 | 不给代码解释"BPE 为何既不 OOV 又能压短序列"、规则表顺序为何重要 |
| 数据划分与泄漏 | assisted | Lesson 08：A/B 对比实验（8 字片段重叠 4 vs 0）；Q1 答出症状（跨文档）未答出根因（块边界只数长度）；Q2「重叠 0 = 无泄漏」答错，经转述/短短语/同源三反例纠正 | 判断一个"先切后分"的划分会不会泄漏，并说出检测器的盲区 |
| softmax 与交叉熵 | assisted 偏 independent | Q2（稀释机制）独立答对；Q1 方向对、数值锚点（loss 地板=0）刚补 | 换 target 后预测 loss 变化 |
| 梯度与参数更新 | assisted | Q1 独立答对（无 step 则参数不动、loss 恒定）；Q2 预测发散命中（lr=2.0 实测 17→1.6e25→inf→nan）；Q3 梯度累加经对照实验讲解 | 用自己的话解释 zero_grad 位置与梯度累积的关系 |
| attention 与 causal mask | unverified | — | — |
| SFT loss mask | unverified | — | — |
| LoRA / DPO / GRPO / KV Cache / 量化 | unverified | — | — |

## 里程碑看板

| 阶段 | 执行状态 | 理解状态 |
|---|---|---|
| 基础过渡 | in_progress（差过拟合实测，留 M3 验证） | assisted |
| M0 诊断与环境 | done（2026-10-18） | assisted |
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

### Lesson 03（2026-10-18）：从零线性分类器（M0 收官）

- 模式：AI 写实现 + 本人阅读验证（本人明确选择"你写我读"）
- 阅读三题：Q1 shape [100,2] 独立答对（"广播=消耗"小误经纠正：广播是复制，矩阵乘才消耗维）；Q2 batch CE 未答出，经"y=一沓标准答案"讲解 + 等值验证（batch 1.373876 == 逐样本平均）补上；Q3 半对（step 命中，zero_ ↔ zero_grad 漏）
- 实验 1（class1 中心 3,3→1.5,1.5）：acc 预测 0.85 vs 实际 0.83 命中；边界点预测"更犹疑"实际更笃定（0.69→0.82）——测试点恰成 class1 正中心，"骑不骑线"由数据分布决定而非点坐标
- 核心教训：acc 天花板由数据可分性决定，不是训练步数（loss 卡 0.35 降不动）；M6 数据标注会遇到同款墙，解法在数据侧
- 初始 loss 0.71 ≈ ln(2) 锚点建立：二分类随机基线
- 决策边界延伸（图解 + 实测）：三步法（代入边界方程比较）独立掌握；"大小 ↔ 哪一侧"初答双向皆反，经中心代入法锚定（0 类中心代入=0，1 类中心代入=3，值近谁判谁）纠正；A/B 点实测 [0.611,0.389] / [0.086,0.914] 验证；附带洞察：笃定程度 = 离线距离 × w 范数（w=训练练出的果断度）
- 脚本：lessons/lesson03_classifier.py

### Lesson 04（2026-10-19）：字符级 tokenizer（M1 预热，轻松课）

- 模式：本人当天下班疲劳，主动选"20 分钟跟读课"；只读代码 + 1 个预测 + 1 行改动，不做推导
- 三预测全错/半错：Q1 猜"6 个数字"实际 22（按**词**数，非按字符数）→ 暴露"词级切分"直觉，正是 tokenizer 核心分岔口（词级：序列短但词表大 + 生词无解；字符级：序列长但词表 = 语料字符集）
- Q2 猜"变成某个数字"实际 KeyError 崩溃 → 字符级词表无兜底；配套观察：22 字符去重后仅 10 个词表项，**字符级毫不压缩**
- Q3 猜"不变"实际数字全变（两套排序两套 id）但解码均还原 → 结论：id 是人为标签、无大小含义，模型必须自学 id 含义（M2 embedding 动机）
- 变式（改 TEXT 为"猫坐在垫子上"）：词表 6、编码 [5,2,1,3,4,0]、解码可还原、`Z` 仍不在词表 → 两预测均命中；本人已理解"换语料 = 换词表 = 换编码"
- 补充教训：assistant 提示语中"这 7 个字"系数错（实际 6 字），已向本人订正
- 下一步（M1 正式课）：byte-level（256 全覆盖，永不 OOV）→ BPE 合并循环 → shifted labels 用在真实文本上
- 脚本：lessons/lesson04_chartokenizer.py

### Lesson 05（2026-10-19）：byte-level 编码（M1 预热续）

- 模式：本人要求直接运行（本轮未做预测，按"直接看输出"处理，不据此升级独立度）
- 结果：英文 22 字符 → 22 token（ASCII 1 字节）；中文 6 字 → 18 token（×3）；"Hello 世界! 🙂" 11 字符 → 18 token
- 单字符开销：'a' 1 字节 [97]，'猫' 3 字节 [231,140,171]，'🙂' 4 字节 [240,159,153,130]
- 永不 OOV 验证：'Z猫🙂ธ\x00' 全部编码成功并还原（含空字节 0）→ "表永远是满的"
- Lesson 04 两短板已治（生字崩、词表随语料漂），但换来序列膨胀：上下文容量下降、attention 计算 O(n²)
- 状态：assisted（本人看过输出与解释，尚未做变式验证）
- 下一步（M1 正式）：BPE 合并循环——从 256 字节出发，合并高频字节对；同时用真实文本做 shifted labels
- 脚本：lessons/lesson05_byte.py

### Lesson 06（2026-10-19）：BPE 合并循环（M1 核心）

- 预测 2/3：Q1「第 1 轮合并 lo」命中（实际 'l'+'o' 16 次 = low6+lower2+lot4+lowest4）；Q3「lowkey 不会崩」命中（实测 ['low','k','e','y']）；Q2「第几轮 low 成整块」答不知，经讲解补上
- 讲解核心：BPE 是"层层搭积木"——每轮只能黏当前相邻两格，但格子本身可以是上轮产物（第 1 轮 lo → 第 2 轮 low）；规则表**有顺序**，顺序即优先级
- 永不 OOV 的机制：能套的规则就套，套不上退回单字符（widest → ['w','i','d','est'] 是被规则覆盖度决定的，不是崩）
- 埋的细节：语料里的 `lot` 是刻意加的，只贡献 lo 不贡献 ow——否则 lo/ow 各 12 次打平，需引入"平手怎么挑"的规则
- 三课收束表：04 字符级（会 OOV、表随语料漂）→ 05 字节级（永不 OOV、序列膨胀 3~4×）→ 06 BPE（永不 OOV、可压回短序列，词表 = 256 + N 条规则）；GPT-2 即 byte-level BPE
- 待做小改动：在训练循环后打印训练完的 splits，观察 ['l','o','w','e','s','t'] → ['low','est'] 的压缩
- 状态：assisted 偏 independent（连续两课预测命中率上升）
- 下一步（M1 剩余）：真实文本 shifted labels 实做、split 规则（先按文档/重复组划分再切 chunk）、成熟库 tokenizer 冻结
- 脚本：lessons/lesson06_bpe.py

### Lesson 07（2026-10-19）：真实文本上的 shifted labels + chunk 切法

- 语料：自造 3 句英文（83 字符），字符级编码，词表 14；token 83 → x = y = 82
- 讲解核心：x 是"读到的"，y 是"该猜的下一个"；硬验证 x[i] == tokens[i+1]
- chunk 机制：每块吃 BLOCK+1=13 个 token 产出 12 组输入/目标（同一段两种错位切法，白赚一份目标）；83 // 13 = 6 块，余 5 个 token 丢弃
- 观察点：块与块不重叠；块是硬切的、不看词边界（"on" 被切成 ... sat | n ...，真实预训练同款行为）
- batch：x/y shape (6,12) → 对应 M0 的 [B,T]；logits 将是 [B,T,V]，每位置对全词表打分，只有 y 对应项是正解
- 3 道解释题待答：①为何 x/y 各 82 而非 83 ②V 应是多少、为什么 ③块 1 开头为何是半截 "n"
- 状态：assisted（Q1 独立答对——"输入没尾巴、预测没头"；Q2 V=词表大小、Q3 块硬切 讲解后补）
- 脚本：lessons/lesson07_shift_labels.py

### Lesson 08（2026-10-19）：split 规则与数据泄漏（M1 概念收官）

- 实验：4 篇文档（doc2/doc3 为逐字相同的重复组），BLOCK=12 / STEP=13，泄漏检测器 = 8 字片段重叠数
- 做法 A（先拼流 → 硬切块 → 按顺序 80/20）：11 块，块 2/5/8 跨文档（'ned muddythe ' 把 doc0 尾与 doc1 头黏在一格）；doc2 前半在 train（块6/7）、尾巴落 val（块8）→ 重叠 **4**
- 做法 B（先按文档/重复组划分 → 再各自切块）：gB→train、gA→val，重叠 **0**
- 关键规矩：重复组必须整体归同一边，否则 val 就是 train 的原题；划分单位选对是根本防线，事后检测只是补救
- Q1 答"train 和 val 都跨两篇文档"——答出症状、未答根因（块边界只由长度决定，切块不可逆，文档边界信息丢失）
- Q2 答"能（B 绝对无泄漏）"——**答错**，经三反例纠正：转述/同义改写（8 字重叠可为 0）、短语短于检测窗口（the cat sat 仅 7 字看不见）、同源批量生成；结论"没检测到 ≠ 不存在"
- 引申：tokenizer 同样只能在 train 文本上训；连到 SPEC 验收标准 #5（用冻结评测证明改善，评测可信度取决于划分与检测覆盖度）
- 状态：assisted（新概念，Q1 半对、Q2 答错已讲解）
- M1 剩余交付物：TinyStories 子集（记许可证/版本/选择原因）+ 成熟库 tokenizer（仅 train 训）+ src/data/ + data/manifest.json
- 脚本：lessons/lesson08_split_leakage.py
