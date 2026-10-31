# Lesson 09：真实语料入场 —— 按「文档」划分 train/val，并写出数据清单（M1 交付物 1/2）
# 运行：.venv/bin/python lessons/lesson09_corpus_split.py
#
# 语料：TinyStories-valid.txt（官方 18.5 MB 英文小故事，许可证 cdla-sharing-1.0）
# 故事之间用 <|endoftext|> 分隔 —— 这既是分隔符，也是真实模型会用的 EOS token

import hashlib
import json
import random
from datetime import date
from pathlib import Path

RAW = Path("data/raw/TinyStories-valid.txt")
OUT = Path("data/processed")
MANIFEST = Path("data/manifest.json")
SEP = "<|endoftext|>"
VAL_RATIO = 0.1
SEED = 1234


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_of_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── 1. 读原始长流，按分隔符切成一篇篇「文档」
text = RAW.read_text(encoding="utf-8")
docs = [d.strip() for d in text.split(SEP)]
docs = [d for d in docs if d]                     # 丢掉空块
lens = [len(d) for d in docs]

print(f"原始文件  {RAW}  {RAW.stat().st_size / 1e6:.1f} MB  {len(text):,} 字符")
print(f"切出文档  {len(docs):,} 篇")
print(f"每篇长度  最短 {min(lens)} / 平均 {sum(lens) // len(lens)} / 最长 {max(lens)} 字符")

# ── 2. ★ 划分单位是「文档」，不是字符位置、也不是 chunk（Lesson 08 的规矩）
rng = random.Random(SEED)
order = list(range(len(docs)))
rng.shuffle(order)
val_idx = set(order[: int(len(docs) * VAL_RATIO)])

train_docs = [d for i, d in enumerate(docs) if i not in val_idx]
val_docs = [d for i, d in enumerate(docs) if i in val_idx]

# ── 3. 重复组检查：按篇划分挡得住"同一篇被劈成两半"，但挡不住"语料里本来就有两篇一样的"
train_hashes = {sha256_of_text(d) for d in train_docs}
val_hashes = {sha256_of_text(d) for d in val_docs}
print(f"\n重复组检查（Lesson 08 的检测器）")
print(f"  train 篇数 {len(train_docs):,}  去重后 {len(train_hashes):,}  → 内部重复 {len(train_docs) - len(train_hashes):,} 篇")
print(f"  val   篇数 {len(val_docs):,}  去重后 {len(val_hashes):,}  → 内部重复 {len(val_docs) - len(val_hashes):,} 篇")
print(f"  train 与 val 逐字相同的篇数：{len(train_hashes & val_hashes):,}")

# ── 4. 落盘：训练/验证文本 + 数据清单（许可证、版本、划分规则、哈希、复现命令）
OUT.mkdir(parents=True, exist_ok=True)
train_path, val_path = OUT / "train.txt", OUT / "val.txt"
train_path.write_text(SEP.join(train_docs), encoding="utf-8")
val_path.write_text(SEP.join(val_docs), encoding="utf-8")

manifest = {
    "dataset": "roneneldan/TinyStories",
    "source_file": "TinyStories-valid.txt",
    "source_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    "download_mirror": "https://hf-mirror.com",
    "license": "cdla-sharing-1.0",
    "downloaded_at": str(date.today()),
    "why_this_corpus": "英文合成小故事，词表极小、句法简单，适合先验证语言建模链路；第一段不以中文能力验收",
    "why_valid_not_train": "官方 train 1.92 GB 超出首个链路需要（1-5M tokens）；本文件 18.5 MB 足够，且划分由项目自己控制",
    "raw": {"bytes": RAW.stat().st_size, "sha256": sha256_of(RAW)},
    "documents": {"total": len(docs), "train": len(train_docs), "val": len(val_docs)},
    "chars": {"total": len(text), "train": sum(len(d) for d in train_docs), "val": sum(len(d) for d in val_docs)},
    "split_rule": f"每篇故事为一个文档；按文档整体随机划分，val_ratio={VAL_RATIO}，seed={SEED}；先划分再切 chunk",
    "duplicate_check": {
        "train_internal_dups": len(train_docs) - len(train_hashes),
        "val_internal_dups": len(val_docs) - len(val_hashes),
        "train_val_exact_dups": len(train_hashes & val_hashes),
    },
    "outputs": {
        str(train_path): sha256_of(train_path),
        str(val_path): sha256_of(val_path),
    },
    "reproduce": ".venv/bin/python lessons/lesson09_corpus_split.py",
}
MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"\n写入  {train_path}  {train_path.stat().st_size / 1e6:.1f} MB")
print(f"写入  {val_path}  {val_path.stat().st_size / 1e6:.1f} MB")
print(f"写入  {MANIFEST}")
print("\n★ 现在手上有了三条文本流：train（可看）/ val（只能考，不许训）/ 原始文件（存档）")
print("★ 下一步（Lesson 10）：BPE 只能在 train.txt 上训 —— 这就是 Lesson 08 那条规矩落地的地方")
