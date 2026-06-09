#!/usr/bin/env python3
"""
build_dataset.py
================
从 HuggingFace 拉取 SGI / SFE / SuperGPQA 三个数据源，归一化 schema，
划分 train/test，并合并输出训练集与测试集。单文件、无外部配置依赖。

划分规则
--------
- SGI-Reasoning : test = 下方内置的 146 个 ID；train = 全集 − test（约 145 条）
- SFE           : 学科分层随机采样，5 学科共 50 train / 100 test，seed=42
- SuperGPQA     : subfield→学科映射后分层随机，124 train / 110 test，seed=42

输出（相对仓库根目录的 data/）
------------------------------
    data/raw/<source>/<split>.json    schema 归一化后的单源 train/test
    data/images/<source>/<id>_<n>.png  从 HF 抽出的图像（SGI / SFE 含图，SuperGPQA 纯文本）
    data/train_combined.json           三源 train 合并
    data/test_combined.json            三源 test 合并
    data/all_combined.json             train + test 合并（部分流水线按全量跑时使用）

用法
----
    python scripts/build_dataset.py                          # 跑全部三源
    python scripts/build_dataset.py --datasets sgi           # 只跑 SGI
    python scripts/build_dataset.py --sfe-language en        # SFE 语言子集（默认 en）
    python scripts/build_dataset.py --no-merge               # 跳过合并步骤

依赖：datasets、Pillow（见仓库 pyproject.toml）。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

# 仓库根目录 = scripts/ 的上一级；数据统一输出到仓库根目录下的 data/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMG_DIR = DATA_DIR / "images"

SEED = 42


# =========================================================================== #
# 学科 taxonomy 映射 + Train/Test 配额
# 所有数据源的 subject 都归一化到 SGI 9 学科。
#   SGI       : 145 train + 146 test
#   SFE       : 50 train + 100 test（5 学科）
#   SuperGPQA : 124 train + 110 test（9 学科，向 SFE 缺失的 4 学科倾斜）
# =========================================================================== #
# SFE 5 学科 → SGI 学科。SFE 不覆盖 Physics / Energy / Information / Neuroscience。
SFE_TO_SGI = {
    "astronomy": "Astronomy",
    "chemistry": "Chemistry",
    "earth": "Earth",
    "earth science": "Earth",
    "earth_science": "Earth",
    "earth sciences": "Earth",
    "life": "Life",
    "life science": "Life",
    "life sciences": "Life",
    "life_science": "Life",
    "biology": "Life",
    "material": "Material",
    "materials": "Material",
    "material science": "Material",
    "materials science": "Material",
    "materials_science": "Material",
}

# SFE 配额：(train, test)，按学科分层
SFE_QUOTA = {
    "Astronomy": (9, 18),
    "Chemistry": (9, 18),
    "Earth": (9, 18),
    "Life": (14, 28),   # SGI 占比最高，略倾斜
    "Material": (9, 18),
}

# SuperGPQA subfield → SGI 学科（关键字启发式，顺序从特异到通用）
SUPERGPQA_KEYWORD_RULES = [
    ("astronomy", lambda s: any(k in s for k in ["astronom", "astrophysic", "cosmolog"]), "Astronomy"),
    ("neuroscience", lambda s: any(k in s for k in ["neuro", "brain", "cognitive sci"]), "Neuroscience"),
    ("energy", lambda s: any(k in s for k in ["energy", "power engineer", "petroleum",
                                              "thermal engineer", "nuclear engineer"]), "Energy"),
    ("material", lambda s: any(k in s for k in ["material", "metallurg", "ceramic"]), "Material"),
    ("information", lambda s: any(k in s for k in ["computer sci", "computer engineer", "electronic",
                                                   "information", "communication engineer", "software"]), "Information"),
    ("earth", lambda s: any(k in s for k in ["geolog", "geophys", "geograph", "atmospheric",
                                             "oceanograph", "meteorolog", "earth sci"]), "Earth"),
    ("life", lambda s: any(k in s for k in ["biolog", "biochem", "genetic", "ecolog", "zoolog",
                                            "botan", "physiolog", "anatom", "microbiolog",
                                            "molecular bio", "cell bio"]), "Life"),
    ("chemistry", lambda s: "chem" in s and "biochem" not in s, "Chemistry"),
    ("physics", lambda s: "physic" in s and "biophysic" not in s, "Physics"),
]

EXCLUDED_SUPERGPQA_KEYWORDS = [
    "law", "education", "philosophy", "literature", "history",
    "art", "military", "agriculture", "economics", "management",
]

# SuperGPQA 配额：向 SFE 缺失的 4 学科（Physics/Energy/Information/Neuroscience）倾斜
SUPERGPQA_QUOTA = {
    "Physics": (22, 17),
    "Energy": (20, 15),
    "Information": (24, 18),
    "Neuroscience": (24, 18),
    "Astronomy": (9, 12),
    "Chemistry": (8, 9),
    "Earth": (8, 9),
    "Life": (9, 8),
    "Material": (0, 4),
}

# 配额自校验
assert sum(t for t, _ in SFE_QUOTA.values()) == 50
assert sum(e for _, e in SFE_QUOTA.values()) == 100
assert sum(t for t, _ in SUPERGPQA_QUOTA.values()) == 124
assert sum(e for _, e in SUPERGPQA_QUOTA.values()) == 110

# SGI test set IDs（146 条）；train = 全集 − 这些 ID
SGI_TEST_IDS = {
    "SGI_Reasoning_0004", "SGI_Reasoning_0005", "SGI_Reasoning_0006", "SGI_Reasoning_0009",
    "SGI_Reasoning_0013", "SGI_Reasoning_0014", "SGI_Reasoning_0018", "SGI_Reasoning_0023",
    "SGI_Reasoning_0025", "SGI_Reasoning_0030", "SGI_Reasoning_0033", "SGI_Reasoning_0036",
    "SGI_Reasoning_0038", "SGI_Reasoning_0039", "SGI_Reasoning_0056", "SGI_Reasoning_0062",
    "SGI_Reasoning_0073", "SGI_Reasoning_0074", "SGI_Reasoning_0079", "SGI_Reasoning_0080",
    "SGI_Reasoning_0088", "SGI_Reasoning_0091", "SGI_Reasoning_0095", "SGI_Reasoning_0096",
    "SGI_Reasoning_0101", "SGI_Reasoning_0106", "SGI_Reasoning_0113", "SGI_Reasoning_0116",
    "SGI_Reasoning_0118", "SGI_Reasoning_0119", "SGI_Reasoning_0121", "SGI_Reasoning_0127",
    "SGI_Reasoning_0128", "SGI_Reasoning_0130", "SGI_Reasoning_0133", "SGI_Reasoning_0140",
    "SGI_Reasoning_0141", "SGI_Reasoning_0144", "SGI_Reasoning_0149", "SGI_Reasoning_0152",
    "SGI_Reasoning_0156", "SGI_Reasoning_0158", "SGI_Reasoning_0159", "SGI_Reasoning_0164",
    "SGI_Reasoning_0170", "SGI_Reasoning_0173", "SGI_Reasoning_0182", "SGI_Reasoning_0183",
    "SGI_Reasoning_0189", "SGI_Reasoning_0193", "SGI_Reasoning_0194", "SGI_Reasoning_0202",
    "SGI_Reasoning_0210", "SGI_Reasoning_0214", "SGI_Reasoning_0216", "SGI_Reasoning_0217",
    "SGI_Reasoning_0225", "SGI_Reasoning_0228", "SGI_Reasoning_0236", "SGI_Reasoning_0237",
    "SGI_Reasoning_0238", "SGI_Reasoning_0241", "SGI_Reasoning_0245", "SGI_Reasoning_0251",
    "SGI_Reasoning_0256", "SGI_Reasoning_0257", "SGI_Reasoning_0263", "SGI_Reasoning_0266",
    "SGI_Reasoning_0267", "SGI_Reasoning_0275", "SGI_Reasoning_0279", "SGI_Reasoning_0289",
    "SGI_Reasoning_0290", "SGI_Reasoning_0012", "SGI_Reasoning_0015", "SGI_Reasoning_0019",
    "SGI_Reasoning_0022", "SGI_Reasoning_0024", "SGI_Reasoning_0035", "SGI_Reasoning_0037",
    "SGI_Reasoning_0042", "SGI_Reasoning_0046", "SGI_Reasoning_0047", "SGI_Reasoning_0049",
    "SGI_Reasoning_0050", "SGI_Reasoning_0051", "SGI_Reasoning_0053", "SGI_Reasoning_0054",
    "SGI_Reasoning_0060", "SGI_Reasoning_0064", "SGI_Reasoning_0070", "SGI_Reasoning_0071",
    "SGI_Reasoning_0077", "SGI_Reasoning_0078", "SGI_Reasoning_0082", "SGI_Reasoning_0092",
    "SGI_Reasoning_0093", "SGI_Reasoning_0099", "SGI_Reasoning_0100", "SGI_Reasoning_0102",
    "SGI_Reasoning_0110", "SGI_Reasoning_0112", "SGI_Reasoning_0115", "SGI_Reasoning_0117",
    "SGI_Reasoning_0122", "SGI_Reasoning_0132", "SGI_Reasoning_0135", "SGI_Reasoning_0139",
    "SGI_Reasoning_0143", "SGI_Reasoning_0150", "SGI_Reasoning_0153", "SGI_Reasoning_0155",
    "SGI_Reasoning_0162", "SGI_Reasoning_0163", "SGI_Reasoning_0172", "SGI_Reasoning_0174",
    "SGI_Reasoning_0178", "SGI_Reasoning_0184", "SGI_Reasoning_0185", "SGI_Reasoning_0191",
    "SGI_Reasoning_0196", "SGI_Reasoning_0198", "SGI_Reasoning_0208", "SGI_Reasoning_0211",
    "SGI_Reasoning_0215", "SGI_Reasoning_0218", "SGI_Reasoning_0221", "SGI_Reasoning_0224",
    "SGI_Reasoning_0226", "SGI_Reasoning_0229", "SGI_Reasoning_0233", "SGI_Reasoning_0239",
    "SGI_Reasoning_0247", "SGI_Reasoning_0248", "SGI_Reasoning_0250", "SGI_Reasoning_0254",
    "SGI_Reasoning_0255", "SGI_Reasoning_0258", "SGI_Reasoning_0260", "SGI_Reasoning_0264",
    "SGI_Reasoning_0268", "SGI_Reasoning_0273", "SGI_Reasoning_0280", "SGI_Reasoning_0283",
    "SGI_Reasoning_0284", "SGI_Reasoning_0288",
}
assert len(SGI_TEST_IDS) == 146, f"SGI test ids 应为 146，实际 {len(SGI_TEST_IDS)}"


def map_supergpqa_subfield(subfield: str) -> str | None:
    """SuperGPQA subfield → SGI subject；不相关返回 None（应被过滤）。"""
    s = (subfield or "").lower().strip()
    if not s:
        return None
    for blocked in EXCLUDED_SUPERGPQA_KEYWORDS:
        if blocked in s:
            return None
    for _, match_fn, sgi_subj in SUPERGPQA_KEYWORD_RULES:
        if match_fn(s):
            return sgi_subj
    return None


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def save_image(pil_image, out_path: Path) -> str:
    """把 PIL Image 存为 PNG，返回绝对路径字符串（便于下游直接读图）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_path, format="PNG")
    return str(out_path.resolve())


def _row_image_to_pil(im) -> object | None:
    """HF 行里的单张图：可能是 PIL、本地路径 str、URL、或 HF dict {bytes, path}。"""
    from PIL import Image

    if im is None:
        return None
    if isinstance(im, Image.Image):
        return im
    if isinstance(im, dict):
        b = im.get("bytes")
        if b:
            return Image.open(BytesIO(b)).convert("RGB")
        p = im.get("path")
        if p:
            return _row_image_to_pil(p)
        return None
    if isinstance(im, (str, Path)):
        s = str(im).strip()
        if not s:
            return None
        if s.startswith(("http://", "https://")):
            with urlopen(s, timeout=120) as resp:
                data = resp.read()
            return Image.open(BytesIO(data)).convert("RGB")
        path = Path(s)
        if path.is_file():
            img = Image.open(path)
            return img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        return None
    return None


def write_json(items: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  ✔ wrote {len(items):4d} items → {out_path.relative_to(ROOT)}")


def normalize_record(
    *,
    source: str,
    record_id: str,
    subject: str,
    original_subject: str,
    split: str,
    question: str,
    options: list | None,
    answer: str | None,
    images: list[str],
    metadata: dict,
) -> dict:
    """统一所有数据集的输出 schema。"""
    return {
        "id": record_id,
        "source": source,
        "subject": subject,
        "original_subject": original_subject,
        "split": split,
        "question": question,
        "options": options,
        "answer": answer,
        "images": images,
        "metadata": metadata,
    }


def _sgi_hf_metadata(row: dict) -> dict:
    """SGI HF 行中可 JSON 化的附加字段；去掉与主表重复列及 PIL 图像/图像列表。"""
    from PIL import Image

    mirrored = {
        "id", "question_id", "subject", "topic", "question", "query", "text",
        "options", "choices", "answer", "gold_answer", "label",
        "image", "images", "figure", "figures", "step_images",
    }
    meta: dict = {}
    for k, v in row.items():
        if k in mirrored:
            continue
        if isinstance(v, Image.Image):
            continue
        if isinstance(v, (list, tuple)) and v and any(isinstance(x, Image.Image) for x in v):
            continue
        meta[k] = v
    return meta


def extract_images(row: dict, source: str, rid: str) -> list[str]:
    """从 HF row 中抽出所有图像并保存到本地，返回路径列表。"""
    image_paths = []
    for img_key in ["images", "image", "figure", "figures"]:
        if img_key in row and row[img_key]:
            imgs = row[img_key] if isinstance(row[img_key], list) else [row[img_key]]
            for n, im in enumerate(imgs):
                pil = _row_image_to_pil(im)
                if pil is None:
                    continue
                out = IMG_DIR / source / f"{rid}_{n}.png"
                image_paths.append(save_image(pil, out))
            return image_paths
    # 兼容 image_1 / image_2 / ... 这种命名
    for n in range(1, 10):
        key = f"image_{n}"
        if key in row and row[key] is not None:
            pil = _row_image_to_pil(row[key])
            if pil is None:
                continue
            out = IMG_DIR / source / f"{rid}_{n-1}.png"
            image_paths.append(save_image(pil, out))
    return image_paths


# --------------------------------------------------------------------------- #
# SGI-Reasoning
# --------------------------------------------------------------------------- #
def process_sgi() -> None:
    """SGI: train = 全集 − SGI_TEST_IDS；test = 内置的 146 IDs。"""
    print("\n[SGI-Reasoning] downloading from InternScience/SGI-Reasoning ...")
    from datasets import load_dataset

    # HF 侧仅有 test split；本地 train/test 由内置 SGI_TEST_IDS 划分
    ds = load_dataset("InternScience/SGI-Reasoning", split="test")
    print(f"  loaded {len(ds)} samples")
    print(f"  schema: {list(ds.features.keys())}")
    print(f"  test_ids: {len(SGI_TEST_IDS)} (expected 146)")

    train_items, test_items = [], []
    for idx, row in enumerate(ds):
        rid = row.get("id") or row.get("question_id") or f"SGI_Reasoning_{idx:04d}"
        subject = row.get("subject") or row.get("topic") or row.get("discipline") or "Unknown"
        question = row.get("question") or row.get("query") or row.get("text") or ""
        options = row.get("options") or row.get("choices")
        answer = row.get("answer") or row.get("gold_answer") or row.get("label")

        image_paths = extract_images(row, "SGI", rid)

        rec = normalize_record(
            source="SGI",
            record_id=rid,
            subject=subject,
            original_subject=subject,
            split="test" if rid in SGI_TEST_IDS else "train",
            question=question,
            options=list(options) if options else None,
            answer=str(answer) if answer is not None else None,
            images=image_paths,
            metadata=_sgi_hf_metadata(row),
        )

        (test_items if rec["split"] == "test" else train_items).append(rec)

    write_json(train_items, RAW_DIR / "SGI" / "train.json")
    write_json(test_items, RAW_DIR / "SGI" / "test.json")
    if len(test_items) != len(SGI_TEST_IDS):
        print(f"  ⚠ test set size mismatch: got {len(test_items)} but expected {len(SGI_TEST_IDS)}")
        missing = SGI_TEST_IDS - {r["id"] for r in test_items}
        if missing:
            print(f"    missing IDs: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")


# --------------------------------------------------------------------------- #
# SFE (Scientists' First Exam)
# --------------------------------------------------------------------------- #
def process_sfe(language: str = "en") -> None:
    """SFE: 学科过滤 → 学科分层随机 → 50 train + 100 test。

    SFE 是双语数据集（en + zh）。默认只取英文子集；schema 无 language 字段则全用。
    """
    print("\n[SFE] downloading from PrismaX/SFE ...")
    from datasets import load_dataset

    # HF 侧当前仅有 test split（全量在 test）；本地 train/test 由脚本内分层采样划分
    ds = load_dataset("PrismaX/SFE", split="test")
    print(f"  loaded {len(ds)} samples")
    print(f"  schema: {list(ds.features.keys())}")

    pool_by_subject = defaultdict(list)
    excluded_lang = 0
    excluded_subject = 0

    for idx, row in enumerate(ds):
        lang = row.get("language") or row.get("lang")
        if lang and isinstance(lang, str) and language != "all":
            if lang.lower() not in {language.lower(), language[:2].lower()}:
                excluded_lang += 1
                continue

        rid = row.get("id") or row.get("question_id") or row.get("uid") or f"SFE_{idx:04d}"

        original_subj = (
            row.get("field") or row.get("discipline") or row.get("subject")
            or row.get("domain") or row.get("category") or ""
        )
        sgi_subj = SFE_TO_SGI.get(original_subj.lower().strip())
        if sgi_subj is None:
            excluded_subject += 1
            continue

        question = (row.get("question") or row.get("query")
                    or row.get("problem") or row.get("text") or "")
        options = row.get("options") or row.get("choices")
        answer = (row.get("answer") or row.get("gold_answer")
                  or row.get("label") or row.get("solution"))

        image_paths = extract_images(row, "SFE", str(rid))

        rec = normalize_record(
            source="SFE",
            record_id=str(rid),
            subject=sgi_subj,
            original_subject=original_subj,
            split="",  # 待填
            question=question,
            options=list(options) if options else None,
            answer=str(answer) if answer is not None else None,
            images=image_paths,
            metadata={
                "language": lang,
                "task": row.get("task"),
                "level": row.get("level") or row.get("cognitive_level"),
                "row_idx": idx,
            },
        )
        pool_by_subject[sgi_subj].append(rec)

    print(f"  filtered: lang_excluded={excluded_lang}, subject_excluded={excluded_subject}")
    print("  per-subject pool size:")
    for s in sorted(pool_by_subject):
        print(f"    {s:14s}: {len(pool_by_subject[s])}")

    # 学科分层随机采样（先 hold out test，再采 train）
    rng = random.Random(SEED)
    train_items, test_items = [], []
    for subject, (n_train, n_test) in SFE_QUOTA.items():
        pool = pool_by_subject.get(subject, [])
        if len(pool) < n_train + n_test:
            print(f"  ⚠ {subject}: pool ({len(pool)}) 小于配额 ({n_train + n_test})，按比例缩放")
            if len(pool) == 0:
                print("      → 检查 SFE_TO_SGI 映射，确认学科名匹配")
                continue
            ratio = len(pool) / (n_train + n_test)
            n_train = max(1, int(n_train * ratio))
            n_test = len(pool) - n_train
        rng.shuffle(pool)
        for r in pool[:n_test]:
            r["split"] = "test"
            test_items.append(r)
        for r in pool[n_test:n_test + n_train]:
            r["split"] = "train"
            train_items.append(r)

    write_json(train_items, RAW_DIR / "SFE" / "train.json")
    write_json(test_items, RAW_DIR / "SFE" / "test.json")


# --------------------------------------------------------------------------- #
# SuperGPQA
# --------------------------------------------------------------------------- #
def process_supergpqa() -> None:
    """SuperGPQA: subfield 映射 → 学科分层随机 → 124 train + 110 test。"""
    print("\n[SuperGPQA] downloading from m-a-p/SuperGPQA ...")
    from datasets import load_dataset

    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    print(f"  loaded {len(ds)} samples")
    print(f"  schema: {list(ds.features.keys())}")

    pool_by_subject = defaultdict(list)
    excluded = 0
    for idx, row in enumerate(ds):
        rid = row.get("uuid") or row.get("id") or row.get("question_id") or f"SGPQA_{idx:06d}"
        subfield = (row.get("subfield") or row.get("sub_field")
                    or row.get("field") or row.get("discipline") or "")
        sgi_subj = map_supergpqa_subfield(subfield)
        if sgi_subj is None:
            excluded += 1
            continue

        question = row.get("question") or row.get("problem") or row.get("query") or ""
        options = row.get("options") or row.get("choices")
        answer = row.get("answer") or row.get("answer_letter") or row.get("gold_answer")

        rec = normalize_record(
            source="SuperGPQA",
            record_id=str(rid),
            subject=sgi_subj,
            original_subject=subfield,
            split="",
            question=question,
            options=list(options) if options else None,
            answer=str(answer) if answer is not None else None,
            images=[],  # 纯文本
            metadata={
                "discipline": row.get("discipline"),
                "field": row.get("field"),
                "subfield": subfield,
                "difficulty": row.get("difficulty"),
            },
        )
        pool_by_subject[sgi_subj].append(rec)

    print(f"  filtered out {excluded}; per-subject pool size:")
    for s in sorted(pool_by_subject):
        print(f"    {s:14s}: {len(pool_by_subject[s])}")

    rng = random.Random(SEED)
    train_items, test_items = [], []
    for subject, (n_train, n_test) in SUPERGPQA_QUOTA.items():
        pool = pool_by_subject.get(subject, [])
        total_needed = n_train + n_test
        if total_needed == 0:
            continue
        if len(pool) < total_needed:
            print(f"  ⚠ {subject}: pool ({len(pool)}) 小于配额 ({total_needed})")
            if len(pool) == 0:
                print("      → 检查 SUPERGPQA_KEYWORD_RULES")
                continue
            ratio = len(pool) / total_needed
            n_train = max(0, int(n_train * ratio))
            n_test = len(pool) - n_train
        rng.shuffle(pool)
        for r in pool[:n_test]:
            r["split"] = "test"
            test_items.append(r)
        for r in pool[n_test:n_test + n_train]:
            r["split"] = "train"
            train_items.append(r)

    write_json(train_items, RAW_DIR / "SuperGPQA" / "train.json")
    write_json(test_items, RAW_DIR / "SuperGPQA" / "test.json")


# --------------------------------------------------------------------------- #
# 合并
# --------------------------------------------------------------------------- #
def _merge(split: str) -> list[dict]:
    combined = []
    for src in ["SGI", "SFE", "SuperGPQA"]:
        path = RAW_DIR / src / f"{split}.json"
        if not path.exists():
            print(f"  ⚠ {path} 不存在，跳过")
            continue
        with open(path) as f:
            items = json.load(f)
        combined.extend(items)
        print(f"  + {src}: {len(items)} items")
    return combined


def merge_all() -> None:
    """合并三源 train / test，并额外输出 all_combined（train + test）。"""
    print("\n[merge] combining train sets from all sources ...")
    train = _merge("train")
    write_json(train, DATA_DIR / "train_combined.json")
    print(f"  total combined train: {len(train)} items (target: ~300)")

    print("\n[merge] combining test sets from all sources ...")
    test = _merge("test")
    write_json(test, DATA_DIR / "test_combined.json")
    print(f"  total combined test: {len(test)} items")

    write_json(train + test, DATA_DIR / "all_combined.json")
    print(f"  total combined all: {len(train) + len(test)} items")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="下载 SGI/SFE/SuperGPQA 并生成 train/test 数据集")
    p.add_argument("--datasets", nargs="+", default=["sgi", "sfe", "supergpqa"],
                   choices=["sgi", "sfe", "supergpqa"])
    p.add_argument("--no-merge", action="store_true", help="跳过 train/test 合并步骤")
    p.add_argument("--sfe-language", default="en", choices=["en", "zh", "all"],
                   help="SFE 语言子集（默认 en）")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if "sgi" in args.datasets:
        process_sgi()
    if "sfe" in args.datasets:
        process_sfe(language=args.sfe_language)
    if "supergpqa" in args.datasets:
        process_supergpqa()
    if not args.no_merge:
        merge_all()

    print("\n[done]")


if __name__ == "__main__":
    main()
