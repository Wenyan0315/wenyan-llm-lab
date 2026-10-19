# Lesson 08：split 规则 —— 为什么必须"先分文档，再切 chunk"
# 目标：亲手制造一次数据泄漏，并看到正确做法把它清零
# 运行：.venv/bin/python lessons/lesson08_split_leakage.py

BLOCK, STEP, N = 12, 13, 8      # 每块 12 个输入 token；实际吃 13 个；用 8 字片段查重叠

# 4 篇文档。第三、四篇是"重复组"——逐字相同的两篇（真实语料里常见：转载、模板、同一来源）
DOCS = [
    ("gB", "rain fell and the road turned muddy"),
    ("gB", "the sun rose over the quiet hill"),
    ("gA", "the cat sat on the mat and the cat was glad"),
    ("gA", "the cat sat on the mat and the cat was glad"),
]


def chunks_of(text):
    """按固定长度硬切：每块覆盖 13 个字符（前 12 个当输入，错位一格出 12 个目标）"""
    return [text[k * STEP:(k + 1) * STEP] for k in range(len(text) // STEP)]


def grams(pieces):
    """把一堆文本块拆成"连续 N 个字"的片段集合"""
    s = set()
    for p in pieces:
        s |= {p[i:i + N] for i in range(len(p) - N + 1)}
    return s


print("语料：")
for i, (g, d) in enumerate(DOCS):
    print(f"  doc{i} [{g}] 长 {len(d):>2}  {d!r}")

# ── 做法 A（错的）：先把所有文档拼成一条长流 → 硬切块 → 再按顺序 80/20 划分
stream = "".join(d for _, d in DOCS)
tags = []                                   # 记录每个字符属于第几篇文档
for i, (_, d) in enumerate(DOCS):
    tags += [i] * len(d)

A_chunks = chunks_of(stream)
cut = int(len(A_chunks) * 0.8)
A_train, A_val = A_chunks[:cut], A_chunks[cut:]

print(f"\n做法 A：拼成长流（{len(stream)} 字符）→ 切出 {len(A_chunks)} 块 → 前 {cut} 块 train，后 {len(A_val)} 块 val")
for k, c in enumerate(A_chunks):
    docs_in = sorted(set(tags[k * STEP:(k + 1) * STEP]))
    where = "train" if k < cut else "val  "
    print(f"  块{k:>2} [{where}] 文档{docs_in}  {c!r}")

# ── 做法 B（对的）：先按文档划分（重复组整体归一边）→ 再各自切块
B_train_text = "".join(d for g, d in DOCS if g == "gB")
B_val_text = "".join(d for g, d in DOCS if g == "gA")
B_train, B_val = chunks_of(B_train_text), chunks_of(B_val_text)

print(f"\n做法 B：先分文档 → gB 共 {len(B_train_text)} 字符进 train，gA 共 {len(B_val_text)} 字符进 val")
print(f"  再各自切块：train {len(B_train)} 块，val {len(B_val)} 块")

# ── 用"8 字片段重叠"当泄漏检测器：val 里有几个片段在 train 里也出现过？
print(f"\n泄漏检测（{N} 字片段重叠数）：")
print(f"  做法 A（先切后分）：{len(grams(A_train) & grams(A_val))}")
print(f"  做法 B（先分后切）：{len(grams(B_train) & grams(B_val))}")

print("\n→ A 的 val 里出现了 train 读过的原文（重复组被劈开、文档被块边界切断）")
print("→ B 的 val 完全来自 train 没见过的文档，重叠归零")
print("\n★ 同一条规矩还管着 tokenizer：BPE 只能在 train 文本上训。")
print("  若让它看过 val 文本，词表就等于「见过」验证集了，仍是泄漏。")
