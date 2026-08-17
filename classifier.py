import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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

    # e5 requires this prefix
    if "e5" in model_name.lower():
        texts = ["passage: " + t for t in texts]

    vecs = model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, vecs)

    return vecs


def run_experiment(train_df, test_df, model_name):
    slug = model_name.replace("/", "_").replace("-", "_")

    X_train = get_embeddings(train_df["text"].tolist(), model_name,
                             cache_path=f"embeddings/train_{slug}.npy")
    X_test = get_embeddings(test_df["text"].tolist(), model_name,
                            cache_path=f"embeddings/test_{slug}.npy")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # three classifiers of increasing complexity
    clfs = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42,
            solver="saga", C=0.1),  # saga works better on high dim embeddings
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
    }

    results = []

    for name, clf in clfs.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        print(f"  F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(classification_report(y_test, preds,
                                    target_names=["benign", "injection"]))

        os.makedirs("models", exist_ok=True)
        with open(f"models/{name}_{slug}.pkl", "wb") as f:
            pickle.dump(clf, f)

        results.append({
            "embedding": model_name,
            "classifier": name,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        })

    return results


train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
print(f"Train: {len(train_df)} | Test: {len(test_df)}\n")

all_results = []

print("Experiment 1: all-MiniLM-L6-v2 (general purpose)")
all_results += run_experiment(train_df, test_df, "all-MiniLM-L6-v2")

print("Experiment 2: intfloat/e5-large-v2 (instruction tuned)")
all_results += run_experiment(train_df, test_df, "intfloat/e5-large-v2")

os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/ablation_results.csv", index=False)
print("\nSaved to results/ablation_results.csv")
print(results_df.to_string(index=False))