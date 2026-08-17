import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


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


def evaluate(clf, vecs, labels, name):
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


train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
train_variants = pd.read_csv("data/processed/train_variants.csv")
test_variants = pd.read_csv("data/processed/test_variants.csv")

print(f"Original train: {len(train_df)} | Train variants: {len(train_variants)}")
print(f"Test variants (eval only, never used in training): {len(test_variants)}")

# add paraphrased variants to training data to improve robustness
augmented = pd.concat(
    [train_df, train_variants[["text", "label", "source", "doc_type"]]],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Augmented train size: {len(augmented)}")

X_train = get_embeddings(augmented["text"].tolist(), "all-MiniLM-L6-v2",
                         cache_path="embeddings/train_augmented_minilm.npy")
X_test = get_embeddings(test_df["text"].tolist(), "all-MiniLM-L6-v2",
                        cache_path="embeddings/test_all_MiniLM_L6_v2.npy")
X_variants = get_embeddings(test_variants["text"].tolist(), "all-MiniLM-L6-v2",
                            cache_path="embeddings/test_variants_minilm.npy")

y_train = augmented["label"].values
y_test = test_df["label"].values
y_variants = test_variants["label"].values

clfs = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="saga", C=0.1),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
}

rows = []

for name, clf in clfs.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open(f"models/{name}_augmented.pkl", "wb") as f:
        pickle.dump(clf, f)

    m1 = evaluate(clf, X_test, y_test, f"{name} — standard test set")
    rows.append({"classifier": name, "training": "augmented",
                 "test_set": "standard", **m1})

    # evaluation: do augmented classifiers generalise to unseen paraphrases?
    m2 = evaluate(clf, X_variants, y_variants,
                  f"{name} — held out paraphrased variants")
    rows.append({"classifier": name, "training": "augmented",
                 "test_set": "paraphrased", **m2})

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv("results/augmented_results.csv", index=False)
print("\nSaved to results/augmented_results.csv")
print(pd.DataFrame(rows).to_string(index=False))