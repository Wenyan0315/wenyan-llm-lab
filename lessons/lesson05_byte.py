# Lesson 05：byte-level 编码 —— 用 256 个字节铺满整张表
# 目标：解决 Lesson 04 的两个短板（遇到生字崩、词表跟着语料漂）
# 运行：.venv/bin/python lessons/lesson05_byte.py

VOCAB_SIZE = 256          # 固定 256，永远是它，不随语料变化

TEXTS = [
    "the cat sat on the mat",
    "猫坐在垫子上",
    "Hello 世界! 🙂",
]

print("=" * 52)
for text in TEXTS:
    raw = text.encode("utf-8")             # 文字 -> 字节串（UTF-8 规则）
    ids = list(raw)                         # 字节串 -> 一串 0~255 的整数
    back = bytes(ids).decode("utf-8")       # 反着来：数字 -> 字节串 -> 文字
    print(f"原文：{text!r}")
    print(f"  字符数 {len(text):>2}   数字个数(token 数) {len(ids):>2}   还原成功 {back == text}")
    print(f"  前 8 个数字：{ids[:8]}")
print("=" * 52)

# ── 关键天平：一个字符到底变成几个数字？
print("\n单个字符的字节开销：")
for ch in ["a", "猫", "🙂"]:
    bs = ch.encode("utf-8")
    print(f"  {ch!r:<8} 占 {len(bs)} 个数字 -> {list(bs)}")

# ── 永不 OOV：随便塞，随便编，都不崩
weird = "Z猫🙂\N{THAI CHARACTER THO THONG}\x00"
ids = list(weird.encode("utf-8"))
print(f"\n乱七八糟的一句：{weird!r}")
print(f"  编码成功，{len(ids)} 个数字：{ids}")
print(f"  还原成功：{bytes(ids).decode('utf-8') == weird}")
print("\n→ 词表固定 256，任何文本都能编码、都能还原；换语料不用换词表。")
