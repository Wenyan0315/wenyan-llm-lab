# Lesson 06：BPE —— 从单字符出发，把高频的"邻居"黏成新 token
# 目标：看懂 BPE 的训练循环（数对 → 挑最高频 → 合并），并用学到的规则去切没见过的词
# 运行：.venv/bin/python lessons/lesson06_bpe.py

from collections import Counter

# ── 0. 语料：把每个词按出现次数重复写出来（×6 就是写 6 遍）
#    真实 BPE 的语料是海量文本，这里只留 6 个词，让每一轮都能看明白
CORPUS = (
    ["low"] * 6 +
    ["lower"] * 2 +
    ["lot"] * 4 +
    ["new"] * 7 +
    ["lowest"] * 4 +
    ["widest"] * 3
)

# 起点：每个词都被拆成一格一格的单字符。
# 这跟 Lesson 05 的 byte-level 是一回事——只不过这里的最小单位是"字符"。
splits = [list(w) for w in CORPUS]
print("起始状态（每个词都是单字符）：")
for w in sorted(set(map(tuple, splits))):
    print("   ", list(w))


# ── 1. 数对：统计所有"相邻的两个格子"各出现了多少次
def count_pairs(splits):
    counts = Counter()
    for word in splits:
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += 1
    return counts


# ── 2. 合并：在一个词里，把所有出现的目标对黏成一格
def merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])   # 两个格子 -> 一个格子
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


# ── 3. 训练循环：数对 → 挑最高频 → 合并 → 记下规则，重复若干轮
merges = []                                     # 学到的规则，(前, 后) 的顺序就是优先级
print("\n训练过程：")
for rnd in range(6):
    counts = count_pairs(splits)
    top3 = counts.most_common(3)
    best, cnt = top3[0]                         # 出现次数最多的那一对
    merges.append(best)
    splits = [merge(w, best) for w in splits]
    print(f"  第 {rnd + 1} 轮：合并 {best[0]!r}+{best[1]!r}（出现 {cnt} 次）  候选前三 {top3}")

print("\n学到的合并规则：")
for i, (a, b) in enumerate(merges):
    print(f"  #{i + 1}  {a!r} + {b!r}  ->  {a + b!r}")


# ── 4. 编码：把学到的规则按顺序套到新文本上
def encode(text):
    pieces = list(text)                         # 第一步永远是"先退化成单字符"
    for pair in merges:                         # 然后按优先级，一条条套规则
        pieces = merge(pieces, pair)
    return pieces


print("\n编码测试（前两个词语料里出现过，后两个没见过）：")
for s in ["lowest", "widest", "lowkey", "newworld"]:
    print(f"  {s!r:>12} -> {encode(s)}")

print("\n→ 没见过的词照样能编码：能套上的规则就套，套不上的退回单字符。永不 OOV。")
