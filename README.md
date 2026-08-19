# Rag-Security-Dissertation
# ML-Based Detection of Indirect Prompt Injection Attacks in RAG Systems

Author: Aarjav Desai (250780404)
Supervisor: Dr Vasileios Klimis
Programme: MSc Artificial Intelligence, Queen Mary University of London

## Overview

This repository contains the full implementation for the dissertation
"ML-Based Detection of Indirect Prompt Injection Attacks in
Retrieval-Augmented Generation Systems: A Robustness Study".

The project trains embedding-based classifiers to detect indirect prompt
injection attacks in RAG pipelines, evaluates how performance degrades
under adversarial paraphrasing, and shows that retraining with
LLM-generated paraphrased variants substantially recovers robustness.

## Files

- dataset.py: Builds the full dataset from BIPIA injection examples and domain-matched benign corpora
- classifier.py: Trains Logistic Regression, XGBoost and MLP on MiniLM and E5 embeddings
- fuzzing.py: Generates paraphrased adversarial variants using llama3.2 via Ollama
- robustness.py: Evaluates baseline classifiers on held-out paraphrased test variants
- augment.py: Retrains classifiers on augmented data and evaluates on held-out variants
- rag_pipeline.py: Full RAG pipeline with block and sanitise defence modes
- judge.py: Measures attack success rate using Claude Haiku as automated LLM-as-judge
- generate_bipia.py: Generates BIPIA injection examples from context and attack files
- models/: Pre-trained classifier models saved as .pkl files
- results/: All experiment results saved as .csv files

## Requirements

Python 3.9 or higher. Install dependencies with:

    pip install sentence-transformers scikit-learn xgboost datasets numpy pandas rank-bm25 anthropic

Ollama must be installed locally with llama3.2 pulled:

    ollama pull llama3.2

BIPIA must be cloned to data/BIPIA/:

    git clone https://github.com/microsoft/BIPIA data/BIPIA

## How to Reproduce Results

Run scripts in this order:

Step 1 - Build the dataset:

    python dataset.py

Downloads Enron emails, CodeSearchNet, AG News and Wikipedia from
HuggingFace automatically. BIPIA must already be cloned first.

Step 2 - Train baseline classifiers:

    python classifier.py

Trains all three classifiers on both MiniLM and E5 embeddings.
E5 takes approximately 6 hours on CPU. MiniLM takes around 14 minutes.

Step 3 - Generate adversarial paraphrased variants:

    python fuzzing.py

Requires Ollama running locally with llama3.2.
Generates 991 training variants and 494 test variants.

Step 4 - Evaluate robustness without augmentation:

    python robustness.py

Tests baseline classifiers on held-out paraphrased test variants.

Step 5 - Adversarial augmentation:

    python augment.py

Retrains classifiers on original data plus paraphrased train variants.
Evaluates on held-out test variants.

Step 6 - Class imbalance analysis:

    python class_imbalance.py

Evaluates at 1:10 injection to benign ratio.

Step 7 - End-to-end RAG pipeline:

    python rag_pipeline.py

Requires Ollama running locally.

Step 8 - LLM-as-judge evaluation:

    python judge.py

Requires Anthropic API key:

    export ANTHROPIC_API_KEY=your_key_here

## Pre-computed Results

All results are in results/:

- ablation_results.csv: MiniLM vs E5 baseline comparison
- robustness_results.csv: Performance under adversarial paraphrasing
- augmented_results.csv: Performance after adversarial augmentation
- judge_results.csv: End-to-end attack success rate by defence mode
- rag_results.csv: RAG pipeline evaluation results

Pre-trained models are in models/ for reproducibility without retraining.

## Key Results

Baseline MiniLM MLP F1: 0.916
Baseline E5 LR F1: 0.967
MLP under paraphrasing (no augmentation): 0.607
MLP after adversarial augmentation: 0.867
MLP at 1:10 class imbalance: 0.913

End-to-end attack success rate:
No defence: 13.0%
Block mode: 0.0%
Sanitise mode: 0.0%

## Note on Data

BIPIA must be cloned separately as shown above. All other datasets
download automatically from HuggingFace when dataset.py is run.
