# Lesson 04：字符级 tokenizer —— 文字怎么变成数字
# 这节课只有一件事：看懂「编码 / 解码」这一步，并亲眼看到它的一个致命短板。
# 运行：.venv/bin/python lessons/lesson04_chartokenizer.py

TEXT = "the cat sat on the mat"

# ── 1. 造词表：把语料里出现过的字符收集起来，去重、排序，编号
chars = sorted(set(TEXT))            # set 去重，sorted 定顺序
vocab_size = len(chars)

print("语料：", TEXT)
print(f"词表大小：{vocab_size}")
print("词表（字符 -> id）：")
for i, ch in enumerate(chars):
    print(f"  {ch!r:>6} -> {i}")

# 两张对照表：一张正向查（字 -> 号），一张反向查（号 -> 字）
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

# ── 2. 编码：文字 -> 一串整数。这一步就是模型真正吃进去的东西
ids = [char_to_id[ch] for ch in TEXT]
print("\n编码结果：", ids)
print("长度（= token 数）：", len(ids), " 原文字符数：", len(TEXT))

# ── 3. 解码：整数 -> 文字。反查表 + join
back = "".join(id_to_char[i] for i in ids)
print("解码还原：", repr(back))
print("往返一致？", back == TEXT)

# ── 4. 关键洞察：id 的大小和顺序是「人为的」，不代表任何含义
# 换一种排序方式重建词表，同一个句子会编出完全不同的数字
rev_chars = sorted(set(TEXT), reverse=True)
rev_map = {ch: i for i, ch in enumerate(rev_chars)}
rev_ids = [rev_map[ch] for ch in TEXT]
print("\n换一种排序后的编码：", rev_ids)
print("解码仍然还原 =", "".join(rev_chars[i] for i in rev_ids) == TEXT)
print("→ 同一句话，两套 id（哪套都没错）。id 只是标签，不是数值含义。")

# ── 5. 致命短板：词表里没有的字，直接查不到
try:
    char_to_id["Z"]                  # 'Z' 在大写字母里，上面语料里没有
except KeyError as e:
    print("\n编码一个词表外的字符：", e)
    print("→ 字符级 tokenizer 遇到没见过的字就崩。这是下一步要解决的问题。")

print("\n词表里到底有哪些字符：", chars)
