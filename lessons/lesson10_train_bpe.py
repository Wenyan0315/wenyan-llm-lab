# Lesson 10：在 train 文本上训练 BPE —— M1 的最后一块（交付物 2/2）
# 运行：.venv/bin/python lessons/lesson10_train_bpe.py
#
# 规矩（Lesson 08）：只在 train.txt 上训 tokenizer，绝不看 val.txt。
# 目标：把 19,791 篇故事压成一条 8K 词表的 token 流，为 M3 预训练准备原料。

import json
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

TRAIN = Path("data/processed/train.txt")
VAL = Path("data/processed/val.txt")
OUT_DIR = Path("data/tokenizer")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEP = "<|endoftext|>"
VOCAB_SIZE = 8000


def read_docs(path):
    return [d for d in path.read_text(encoding="utf-8").split(SEP) if d.strip()]


train_docs = read_docs(TRAIN)
val_docs = read_docs(VAL)

# ── 1. 搭一个 byte-level BPE：先按字节拆（永不言 OOV），再把高频字节对合并
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[SEP],                              # 整串当一个 token，不参与字节拆分
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 个字节全放进来 = 永不 OOV 的地基
    show_progress=False,
)

print(f"训练语料：{len(train_docs):,} 篇（只来自 train.txt）")
t0 = time.time()
tokenizer.train_from_iterator(train_docs, trainer=trainer)
print(f"训练耗时：{time.time() - t0:.1f} 秒")
print(f"词表大小：{tokenizer.get_vocab_size():,}")

tokenizer.save(str(OUT_DIR / "bpe8k.json"))

# ── 2. 三个必须验证的东西
sample = "Once upon a time, there was a little girl named Lily. She loved to play in the garden."
ids = tokenizer.encode(sample).ids
print(f"\n[往返] 原文 == 解码：{tokenizer.decode(ids) == sample}")
print(f"[往返] 词数 {len(sample.split())}  字符 {len(sample)}  token {len(ids)}")
print(f"[往返] 前 12 个 token 及还原：{[(i, tokenizer.decode([i])) for i in ids[:12]]}")

sep_ids = tokenizer.encode(SEP).ids
print(f"\n[EOS] '{SEP}' → {sep_ids}  （{'1 个 token ✓' if len(sep_ids) == 1 else '不是 1 个 ✗'}）")

cn = "猫坐在垫子上，看着窗外的雨。"
cn_ids = tokenizer.encode(cn).ids
print(f"[跨语言] 中文往返成功：{tokenizer.decode(cn_ids) == cn}  字符 {len(cn)} → token {len(cn_ids)}")

# ── 3. 三种编码的序列长度对比（SPEC 要求：同一文本、不同词表，只比长度不比 perplexity）
print("\n三种编码，同一段文本的序列长度：")
print(f"{'文本':<28}{'字符':>6}{'字节':>6}{'BPE':>7}{'字符/token':>12}")
for name, text in [
    ("英文样本", sample),
    ("中文样本", cn),
    ("长英文段落", sample * 3),
]:
    n_char, n_byte = len(text), len(text.encode("utf-8"))
    n_bpe = len(tokenizer.encode(text).ids)
    print(f"{name:<28}{n_char:>6}{n_byte:>6}{n_bpe:>7}{n_char / n_bpe:>11.2f}×")

# ── 4. 量化交付：整条 train / val 流各有多少 token
t0 = time.time()
train_tokens = sum(len(e.ids) for e in tokenizer.encode_batch(train_docs))
val_tokens = sum(len(e.ids) for e in tokenizer.encode_batch(val_docs))
print(f"\n全量编码耗时：{time.time() - t0:.1f} 秒")
print(f"train：{len(train_docs):,} 篇 → {train_tokens:,} tokens")
print(f"val  ：{len(val_docs):,} 篇 → {val_tokens:,} tokens")
print(f"压缩比：{sum(len(d) for d in train_docs) / train_tokens:.2f} 字符/token")

# ── 5. 把 tokenizer 信息补进数据清单
manifest_path = Path("data/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["tokenizer"] = {
    "type": "byte-level BPE",
    "library": "tokenizers",
    "vocab_size": tokenizer.get_vocab_size(),
    "special_tokens": [SEP],
    "trained_on": "data/processed/train.txt only（未使用 val 文本）",
    "output": str(OUT_DIR / "bpe8k.json"),
    "train_tokens": train_tokens,
    "val_tokens": val_tokens,
    "chars_per_token": round(sum(len(d) for d in train_docs) / train_tokens, 3),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"\n写入 {manifest_path}（tokenizer 段已追加）")
print("\n★ M1 交付完成：数据划分 + tokenizer + 数据清单。M3 的原料已就位。")
