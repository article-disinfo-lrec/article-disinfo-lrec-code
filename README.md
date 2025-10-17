# What Signals Really Matter for Misinformation Tasks? <br> Evaluating Fake-News Detection and Virality Prediction under Real-World Constraints

This repository contains the codebase for an article studying automatic classification of (a) fake vs. real news and (b) viral vs. non‑viral items / propagations across two different news datasets.


## 1. Tasks

| Task | Definition | Label Source |
|------|------------|--------------|
| Disinformation Detection | Binary classification: fake vs real news item / propagation | Dataset ground‑truth labels (Evons & FakeNewsNet-Politifact) |
| Virality Prediction | Binary classification: will an item / propagation be “viral” | Evons: ≥ 95th percentile Facebook engagement. FakeNewsNet: ≥ median total likes across propagations |

## 2. Datasets

### Evons Dataset
Static news articles with metadata and engagement statistics (e.g. Facebook). Modeling treats each article independently (non‑sequential). Text = title + caption/description. Virality label derived from high‑end engagement threshold (95th percentile). See `evons/data/readme.md` for download the data.

### FakeNewsNet (Politifact subset)
Twitter propagation trees. Each propagation becomes a sequence of per‑tweet features: text embedding + scalar metadata (verification flag, follower/following counts, favorites, elapsed time, etc.). Virality defined via median total likes threshold. To know how to access data, see `FakeNewsNet/data/readme.md`.

## 3. Repository Overview

```
memoire_disinfo/
├── evons/
│   ├── data/                        # Evons raw data & precomputed embeddings (external download)
│   ├── disinformation_detection/    # MLP variants (RoBERTa & Mistral)
│   └── virality_prediction/         # MLP baseline + source / engagement feature variants
│
├── FakeNewsNet/
│   ├── data/                        # Politifact sequences & embeddings (external download)
│   ├── data_preprocessing/          # Scripts: data retrieval, path creation, embedding generation
│   ├── disinformation_detection/    # Sequence model notebooks (CNN/RNN/GRU/LSTM/Transformer)
│   └── virality_prediction/         # Same architectures for virality label
```

Both dataset folders intentionally mirror a two‑task layout for clarity and comparability.


## 4. Data Access & Privacy

Data are **not** committed because of size & licensing:
* Evons: follow instructions in folder
* FakeNewsNet Politifact: raw data [available here](https://github.com/KaiDMML/FakeNewsNet); processed and anonymized sequences via protected Drive link. 
See each dataset’s `data/readme.md` for more information.


## 5. How To Reproduce

1. Obtain & place data as per `evons/data/readme.md` and `FakeNewsNet/data/readme.md`.

2. For **FakeNewsNet**: Use scripts in `FakeNewsNet/data_preprocessing/`. <br>
If starting from the complete dataset, run: `path_creation.py`, then `create_embeddings.py` or `create_embeddings_mistral.py`. <br>
If starting from anonymized, already processed data, run: `rebuild_dataset.py`, then `create_embeddings.py` or `create_embeddings_mistral.py`.
<br>**Evons** notebook already provide code for embedding texts on-the-fly if not available in `evons/data` folder. 
3. Open the relevant notebook (e.g., `evons/disinformation_detection/MLP.ipynb`) and execute cells top‑to‑bottom. Notebooks are self‑contained (data paths assume relative placement inside each dataset’s `data/`).
4. Compare output metrics across variants.

### Suggested Python Environment 
The following packages are required:
`torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `seaborn`, `wandb`. 

`mistralai` required for Mistral embedding generation.

## 6. Results

Below are the main performance tables reported in the paper (10-fold cross-validation):

### EVONS — Fake News Detection

| Model | Acc | BalAcc | F1 | Prec | Rec | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | **0.991** | **0.991** | **0.990** | **0.990** | **0.990** | **0.999** |
| Logistic Regression | 0.972 | 0.971 | 0.969 | 0.971 | 0.967 | 0.996 |
| Random Forest | 0.932 | 0.931 | 0.926 | 0.925 | 0.927 | 0.983 |
| Dummy (stratified) | 0.501 | 0.498 | 0.457 | 0.458 | 0.456 | 0.501 |

### EVONS — Virality Prediction (95th percentile label)

| Model | Acc | BalAcc | F1 | Prec | Rec | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.885 | 0.712 | 0.312 | 0.224 | 0.519 | 0.842 |
| Source embedding | 0.865 | **0.756** | 0.322 | 0.217 | **0.635** | **0.869** |
| Avg. engagement | 0.869 | 0.751 | 0.322 | 0.218 | 0.621 | 0.867 |
| Gating | 0.869 | 0.751 | **0.323** | **0.219** | 0.620 | 0.868 |
| Logistic Regression | 0.761 | 0.783 | 0.252 | 0.150 | 0.807 | 0.866 |
| Random Forest | 0.855 | 0.699 | 0.266 | 0.178 | 0.526 | 0.811 |
| Dummy (stratified) | **0.905** | 0.500 | 0.049 | 0.049 | 0.049 | 0.501 |

### EVONS — Virality Prediction (Fβ, β large)

| Model | Acc | BalAcc | F1 | Prec | Rec | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.688 | 0.772 | 0.219 | 0.126 | **0.865** | 0.853 |
| Source embedding | 0.740 | 0.787 | 0.245 | 0.144 | 0.839 | 0.868 |
| Avg. engagement | 0.728 | 0.784 | 0.239 | 0.139 | 0.847 | 0.867 |
| Gating | **0.755** | **0.789** | **0.253** | **0.149** | 0.828 | **0.870** |

### FakeNewsNet — Fake News Detection

| Model | Acc | BalAcc | F1 | Prec | Rec | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Transformer encoder | **0.945** | 0.927 | **0.906** | 0.933 | 0.883 | 0.965 |
| GRU | 0.935 | 0.918 | 0.891 | 0.912 | 0.874 | 0.961 |
| RNN | 0.941 | 0.926 | 0.901 | 0.919 | 0.886 | 0.963 |
| LSTM | 0.936 | 0.916 | 0.891 | 0.921 | 0.866 | 0.963 |
| CNN | 0.928 | 0.904 | 0.876 | 0.912 | 0.846 | 0.962 |
| Logistic Regression | 0.939 | **0.929** | 0.899 | 0.896 | **0.906** | **0.971** |
| Random Forest | 0.920 | 0.893 | 0.861 | 0.902 | 0.826 | 0.956 |
| Dummy (stratified) | 0.578 | 0.499 | 0.300 | 0.300 | 0.300 | 0.499 |

### FakeNewsNet — Virality Prediction (median split)

| Model | Acc | BalAcc | F1 | Prec | Rec | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Transformer encoder | **0.776** | **0.776** | **0.798** | 0.737 | **0.886** | **0.837** |
| GRU | 0.768 | 0.768 | 0.793 | 0.720 | **0.886** | 0.823 |
| RNN | 0.762 | 0.762 | 0.787 | 0.719 | 0.871 | 0.818 |
| LSTM | 0.766 | 0.766 | 0.789 | 0.725 | 0.869 | 0.820 |
| CNN | 0.770 | 0.770 | 0.790 | **0.731** | 0.864 | 0.830 |
| Random Forest | 0.743 | 0.743 | 0.769 | 0.701 | 0.853 | 0.808 |
| Logistic Regression | 0.691 | 0.691 | 0.695 | 0.687 | 0.705 | 0.768 |
| Dummy (stratified) | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 |

## 7. Folder Quick Reference

| Path | Purpose |
|------|---------|
| `evons/disinformation_detection/` | Article fake vs real (MLP variants) |
| `evons/virality_prediction/` | Article virality (MLP + feature fusion variants) |
| `FakeNewsNet/data_preprocessing/` | Build tweet sequences & embeddings |
| `FakeNewsNet/disinformation_detection/` | Propagation fake vs real (sequence models) |
| `FakeNewsNet/virality_prediction/` | Propagation virality (sequence models) |


---
**Important**: the code does not save model weights after training. To save them, modify the notebooks, specifically the 
_`train_single_fold`_ functions.

---
For questions or access issues (e.g., processed data links), contact the author.