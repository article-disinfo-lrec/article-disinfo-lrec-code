# Disinformation detection - Evons

This folder contains two experiment notebooks that train and evaluate a lightweight multilayer perceptron (MLP) binary classifier to detect fake news on the Evons dataset. 

Both workflows use stratified k-fold cross-validation to obtain robust performance estimates.

- `MLP.ipynb` – MLP using RoBERTa embeddings: Generates (or loads) 768-dim [CLS] embeddings separately for articles' titles and captions with roberta-base, concatenates them (1536-dim), trains an MLP head with class-weighted loss, and additionally benchmarks classic ML baselines (dummy, logistic regression, random forest) on the same features.

- `MLP_mistral.ipynb` – MLP using Mistral embeddings: Loads higher-dimensional precomputed Mistral embeddings for title and description (concatenated size larger than RoBERTa), applies the same cross-validation training loop and logging. No baselines are trained.