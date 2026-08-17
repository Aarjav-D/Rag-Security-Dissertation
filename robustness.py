import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def get_embeddings(texts, model_name, batch_size=32, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return np.load(cache_path)
    print(f"  Embedding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    if "e5" in model_name.lower():
        texts = ["passage: " + t for t in texts]
    vecs = model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, vecs)
    return vecs


def test_classifier(clf, vecs, labels, name):
    preds = clf.predict(vecs)
    print(f"\n{name}")
    print(classification_report(labels, preds,
                                target_names=["benign", "injection"],
                                labels=[0, 1]))
    return {
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4)
    }


# test the baseline classifiers on paraphrased variants they never saw during training
# this shows how much performance drops when attackers rephrase their injections
variants = pd.read_csv("data/processed/test_variants.csv")
print(f"Test variants: {len(variants)}")

labels = variants["label"].values
slug = "all_MiniLM_L6_v2"

vecs = get_embeddings(variants["text"].tolist(), "all-MiniLM-L6-v2",
                      cache_path="embeddings/test_variants_minilm.npy")

rows = []
for name in ["logistic_regression", "xgboost", "mlp"]:
    path = f"models/{name}_{slug}.pkl"
    if not os.path.exists(path):
        print(f"Model not found: {path}, skipping")
        continue
    with open(path, "rb") as f:
        clf = pickle.load(f)
    metrics = test_classifier(clf, vecs, labels, f"{name} on paraphrased variants")
    rows.append({"classifier": name, "embedding": "MiniLM",
                 "test_set": "paraphrased", **metrics})

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/robustness_results.csv", index=False)
print("\nSaved to results/robustness_results.csv")
print(pd.DataFrame(rows).to_string(index=False))