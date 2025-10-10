# Article on disinformation detection and virality prediction - Temporary Readme

This repository contains the codebase for an article studying automatic classification of (a) fake vs. real news and (b) viral vs. non‑viral items / propagations across two different news datasets. [DRAFT INCIPIT]


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

## 6. Folder Quick Reference

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