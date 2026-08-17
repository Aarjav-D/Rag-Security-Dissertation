import warnings
warnings.filterwarnings("ignore")

import json
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_bipia_full(bipia_path="data/BIPIA/benchmark"):
    # keep bipia train and test splits separate
    # this prevents any leakage later
    print("Loading BIPIA")
    train_records = []
    test_records = []

    text_tasks = ["email", "table"]
    for task in text_tasks:
        for split, records_list in [("train", train_records), ("test", test_records)]:
            ctx_file = os.path.join(bipia_path, task, f"{split}.jsonl")
            atk_file = os.path.join(bipia_path, f"text_attack_{split}.json")
            if not os.path.exists(ctx_file) or not os.path.exists(atk_file):
                continue
            with open(ctx_file) as f:
                contexts = [json.loads(l) for l in f if l.strip()]
            with open(atk_file) as f:
                attacks = json.load(f)
            all_attacks = [instr for instrs in attacks.values() for instr in instrs]
            for ctx in contexts:
                base = ctx.get("context", ctx.get("passage", ""))
                if isinstance(base, list):
                    base = " ".join(str(x) for x in base)
                if not base:
                    continue
                for atk in all_attacks:
                    records_list.append({
                        "text": (base[:800] + "\n\n" + atk)[:1000],
                        "label": 1,
                        "source": "bipia",
                        "doc_type": task
                    })

    for split, records_list in [("train", train_records), ("test", test_records)]:
        ctx_file = os.path.join(bipia_path, "code", f"{split}.jsonl")
        atk_file = os.path.join(bipia_path, f"code_attack_{split}.json")
        if not os.path.exists(ctx_file) or not os.path.exists(atk_file):
            continue
        with open(ctx_file) as f:
            contexts = [json.loads(l) for l in f if l.strip()]
        with open(atk_file) as f:
            attacks = json.load(f)
        all_attacks = [instr for instrs in attacks.values() for instr in instrs]
        for ctx in contexts:
            base = ctx.get("context", ctx.get("passage", ""))
            if isinstance(base, list):
                base = " ".join(str(x) for x in base)
            if not base:
                continue
            for atk in all_attacks:
                records_list.append({
                    "text": (base[:800] + "\n\n" + atk)[:1000],
                    "label": 1,
                    "source": "bipia",
                    "doc_type": "code"
                })

    print(f"  Train: {len(train_records)} | Test: {len(test_records)} BIPIA examples")
    return train_records, test_records


def load_enron(limit=30000):
    # domain matched benign emails to pair with bipia email injections
    print("Loading Enron...")
    ds = load_dataset("SetFit/enron_spam", split="train")
    records = []
    seen = set()
    for ex in ds:
        text = ex.get("text", "")
        if not text or text in seen or len(text) < 50:
            continue
        if ex.get("label", -1) != 0:
            continue
        seen.add(text)
        records.append({
            "text": text[:1000],
            "label": 0,
            "source": "enron",
            "doc_type": "email"
        })
        if len(records) >= limit:
            break
    print(f"  {len(records)} examples")
    return records


def load_code(limit=30000):
    # python docstrings as benign counterpart to bipia code injections
    print("Loading CodeSearchNet...")
    ds = load_dataset("code_search_net", "python", split="train")
    records = []
    seen = set()
    for ex in ds:
        text = ex.get("func_documentation_string", "")
        if not text or text in seen or len(text) < 30:
            continue
        seen.add(text)
        records.append({
            "text": text[:1000],
            "label": 0,
            "source": "codesearchnet",
            "doc_type": "code"
        })
        if len(records) >= limit:
            break
    print(f"  {len(records)} examples")
    return records


def load_news(limit=30000):
    print("Loading AG News...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    records = []
    seen = set()
    for ex in ds:
        text = ex.get("text", "")
        if not text or text in seen or len(text) < 50:
            continue
        seen.add(text)
        records.append({
            "text": text[:1000],
            "label": 0,
            "source": "agnews",
            "doc_type": "news"
        })
        if len(records) >= limit:
            break
    print(f"  {len(records)} examples")
    return records


def load_wiki(limit=20000):
    print("Loading Wikipedia")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    records = []
    seen = set()
    for ex in ds:
        text = ex.get("text", "").split("\n\n")[0]  # first paragraph only
        if not text or text in seen or len(text) < 100:
            continue
        seen.add(text)
        records.append({
            "text": text[:1000],
            "label": 0,
            "source": "wikipedia",
            "doc_type": "wiki"
        })
        if len(records) >= limit:
            break
    print(f"  {len(records)} examples")
    return records


def build_dataset():
    bipia_train, bipia_test = load_bipia_full()
    benign = load_enron() + load_code() + load_news() + load_wiki()

    print(f"\nBIPA train: {len(bipia_train)} | BIPIA test: {len(bipia_test)}")
    print(f"Benign total: {len(benign)}")

    # split benign 80/20 separately then combine with bipia splits
    benign_df = pd.DataFrame(benign)
    benign_df["text"] = benign_df["text"].astype(str)
    benign_df = benign_df.drop_duplicates(subset="text").reset_index(drop=True)

    benign_train, benign_test = train_test_split(
        benign_df, test_size=0.2, random_state=42)

    train = pd.concat([
        pd.DataFrame(bipia_train), benign_train
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    test = pd.concat([
        pd.DataFrame(bipia_test), benign_test
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    train["text"] = train["text"].astype(str)
    test["text"] = test["text"].astype(str)

    # remove duplicates within each split
    train = train.drop_duplicates(subset="text").reset_index(drop=True)
    test = test.drop_duplicates(subset="text").reset_index(drop=True)

    # critical check - no text should appear in both splits
    overlap = set(train["text"]) & set(test["text"])
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts!"

    n_inj = train["label"].sum()
    n_ben = (train["label"] == 0).sum()
    print(f"\nTrain: {len(train)} ({n_inj} inj, {n_ben} ben) | Ratio: 1:{round(n_ben/n_inj, 1)}")
    print(f"Test: {len(test)} ({test['label'].sum()} inj, {(test['label']==0).sum()} ben)")

    return train, test


train_df, test_df = build_dataset()

os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nBreakdown by source:")
print(train_df["source"].value_counts())
print("\nSample injection:")
print(train_df[train_df["label"]==1]["text"].iloc[0][:300])
print("\nSample benign:")
print(train_df[train_df["label"]==0]["text"].iloc[0][:200])