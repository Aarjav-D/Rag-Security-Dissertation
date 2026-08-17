import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier


def get_embeddings(texts, model_name="all-MiniLM-L6-v2", cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return np.load(cache_path)
    print(f"  Embedding {len(texts)} texts...")
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True,
                        convert_to_numpy=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, vecs)
    return vecs


def get_clfs():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="saga", C=0.1),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    }


def run(train_inj, train_ben, test_inj, test_ben, name, cache_slug):
    # combine injection and benign for train and test
    train = pd.concat([train_inj, train_ben]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat([test_inj, test_ben]).sample(
        frac=1, random_state=42).reset_index(drop=True)

    print(f"\n{name}")
    print(f"  Train: {len(train_inj)} inj + {len(train_ben)} ben")
    print(f"  Test:  {len(test_inj)} inj + {len(test_ben)} ben")

    X_train = get_embeddings(train["text"].tolist(),
                             cache_path=f"embeddings/cross_{cache_slug}_train.npy")
    X_test = get_embeddings(test["text"].tolist(),
                            cache_path=f"embeddings/cross_{cache_slug}_test.npy")
    y_train = train["label"].values
    y_test = test["label"].values

    rows = []
    for clf_name, clf in get_clfs().items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print(f"  {clf_name}: F1 {f1_score(y_test, preds, zero_division=0):.4f}")
        rows.append({
            "experiment": name,
            "classifier": clf_name,
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4)
        })
    return rows


train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# split by source
train_bipia = train_df[train_df["source"] == "bipia"]
train_hacka = train_df[train_df["source"] == "hackaprompt"]
test_bipia = test_df[test_df["source"] == "bipia"]
test_hacka = test_df[test_df["source"] == "hackaprompt"]

# benign pool - same for all experiments
train_ben = train_df[train_df["label"] == 0]
test_ben = test_df[test_df["label"] == 0]

print(f"BIPIA - Train: {len(train_bipia)} | Test: {len(test_bipia)}")
print(f"HackAPrompt - Train: {len(train_hacka)} | Test: {len(test_hacka)}")

all_rows = []

# within-source: train and test on same injection type
all_rows += run(train_bipia, train_ben.sample(len(train_bipia)*5, random_state=42, replace=True),
                test_bipia, test_ben.sample(len(test_bipia)*5, random_state=42, replace=True),
                "BIPIA → BIPIA", "bipia_bipia")

all_rows += run(train_hacka, train_ben.sample(min(len(train_hacka)*5, len(train_ben)), random_state=42),
                test_hacka, test_ben.sample(min(len(test_hacka)*5, len(test_ben)), random_state=42),
                "HackAPrompt → HackAPrompt", "hacka_hacka")

# cross-source: train on one, test on the other
# this tests whether the classifier learns general injection patterns
# or just overfits to one dataset's style
all_rows += run(train_bipia, train_ben.sample(len(train_bipia)*5, random_state=42, replace=True),
                test_hacka, test_ben.sample(min(len(test_hacka)*5, len(test_ben)), random_state=42),
                "BIPIA → HackAPrompt", "bipia_hacka")

all_rows += run(train_hacka, train_ben.sample(min(len(train_hacka)*5, len(train_ben)), random_state=42),
                test_bipia, test_ben.sample(len(test_bipia)*5, random_state=42, replace=True),
                "HackAPrompt → BIPIA", "hacka_bipia")

# train on both, test on each separately
train_both = pd.concat([train_bipia, train_hacka])
n_both = min(len(train_both)*5, len(train_ben))

all_rows += run(train_both, train_ben.sample(n_both, random_state=42),
                test_bipia, test_ben.sample(len(test_bipia)*5, random_state=42, replace=True),
                "Both → BIPIA", "both_bipia")

all_rows += run(train_both, train_ben.sample(n_both, random_state=42),
                test_hacka, test_ben.sample(min(len(test_hacka)*5, len(test_ben)), random_state=42),
                "Both → HackAPrompt", "both_hacka")

os.makedirs("results", exist_ok=True)
pd.DataFrame(all_rows).to_csv("results/cross_source_results.csv", index=False)
print("\nSaved to results/cross_source_results.csv")
print(pd.DataFrame(all_rows).to_string(index=False))