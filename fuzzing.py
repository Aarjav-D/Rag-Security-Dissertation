import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import requests
import os
import time
from tqdm import tqdm


def is_valid(original, result):
    refusals = ["i can't", "i cannot", "i'm unable", "as an ai",
                "i don't", "i won't", "i apologize", "i'm sorry"]
    if any(p in result.lower() for p in refusals):
        return False
    if len(result) < 15:
        return False
    if result.strip() == original.strip():
        return False
    # reject if too similar - over 90% word overlap means barely changed
    orig_words = set(original.lower().split())
    para_words = set(result.lower().split())
    if len(orig_words) > 0 and len(orig_words & para_words) / len(orig_words) > 0.9:
        return False
    return True


def can_paraphrase(text):
    # llama3.2 cant handle table examples with these chars
    if any(c in text for c in ["˄", "˅", "|", "──", "┌", "┐"]):
        return False
    if len(text.split()) < 5:
        return False
    return True


def paraphrase(text, max_retries=3):
    prompt = f"""Rewrite the following text using completely different words and sentence structure.
Keep the exact same meaning but express it differently.
Do not copy phrases from the original.
Output only the rewritten text, nothing else.

Text: {text}

Rewritten:"""

    for _ in range(max_retries):
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.8}},
                timeout=120
            )
            result = r.json().get("response", "").strip()
            if is_valid(text, result):
                return result
        except Exception:
            time.sleep(2)
    return None


def generate_variants(injections, num_samples, save_path):
    # resume from where we left off if interrupted
    if os.path.exists(save_path):
        done = pd.read_csv(save_path)
        done_originals = set(done["original"].tolist())
        print(f"Resuming — {len(done)} already saved")
    else:
        done_originals = set()

    candidates = injections[
        injections["text"].apply(can_paraphrase)
    ].reset_index(drop=True)

    print(f"Paraphraseable: {len(candidates)}/{len(injections)}")

    todo = candidates[
        ~candidates["text"].isin(done_originals)
    ].head(num_samples - len(done_originals))

    print(f"Generating {len(todo)} variants...")
    failed = 0

    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        result = paraphrase(row["text"])
        if result:
            new_row = pd.DataFrame([{
                "text": result,
                "label": 1,
                "source": "paraphrased",
                "doc_type": row.get("doc_type", "unknown"),
                "original": row["text"]
            }])
            header = not os.path.exists(save_path)
            new_row.to_csv(save_path, mode="a", header=header, index=False)
        else:
            failed += 1

    print(f"Done | Failed: {failed}")
    return pd.read_csv(save_path)


train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

train_inj = train_df[train_df["label"] == 1].reset_index(drop=True)
test_inj = test_df[test_df["label"] == 1].reset_index(drop=True)

print(f"Train injections: {len(train_inj)} | Test injections: {len(test_inj)}")

# train variants go into augmentation, test variants go into evaluation only
# keeping them separate is critical to avoid leakage
print("\nGenerating train variants")
train_variants = generate_variants(train_inj, 1000, "data/processed/train_variants.csv")
print(f"Train variants: {len(train_variants)}")

print("\nGenerating test variants")
test_variants = generate_variants(test_inj, 500, "data/processed/test_variants.csv")
print(f"Test variants: {len(test_variants)}")

overlap = set(train_variants["original"]) & set(test_variants["original"])
print(f"\nLeakage check: {len(overlap)} originals in both splits")

print("\nSample:")
print("Original:", train_variants["original"].iloc[0][:150])
print("Paraphrased:", train_variants["text"].iloc[0][:150])