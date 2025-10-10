# Disinformation detection - FakeNewsNet

These notebooks implement and compare sequence classification models for detecting fake vs. real news propagation on Twitter. Each constructs per-propagation sequences by pairing precomputed text embeddings with lightweight per‑tweet scalar metadata (verification flag, follower/following counts, favorites, elapsed time) and performs stratified cross‑validation.

Model variants:

`CNN_model_detection.ipynb`: Concatenates BERT embeddings with embedded numerical features per tweet, applies stacked 1D convolutions over the short sequence, adaptive max‑pools across time, then uses a dense head for binary prediction. Captures local temporal/positional patterns via convolutional receptive fields instead of recurrent state.

`RNN_model_detection.ipynb`: Uses a bidirectional vanilla RNN over fused (BERT + numerical) per‑tweet representations; final time step hidden state feeds a classification head. Provides a simple recurrent baseline emphasizing sequential dependence without gating.

`GRU_model_detection.ipynb`: Replaces plain RNN with a bidirectional GRU for gated temporal state aggregation. Includes two ablation variants: (a) GRU without BERT (only numerical metadata embeddings) and (b) GRU without numerical features (only BERT), isolating contribution of each feature group. Also adds classical baselines (dummy, logistic regression, random forest) using aggregated sequence statistics.

`LSTM_model_detection.ipynb`: Employs a bidirectional LSTM on the fused representations, leveraging input/forget/output gates for potentially richer long(er) dependency modeling across the limited sequence window.

`transformers_model_detection.ipynb`: Projects concatenated BERT + numerical embeddings into a shared d_model space, adds learned positional embeddings, and passes through a multi-layer pre‑norm Transformer encoder. Applies masked max pooling across time (respecting true lengths) before a layer‑normalized feedforward head, emphasizing parallel self‑attention over recurrent accumulation.

`GRU_model_detection_mistral.ipynb`: Mirrors the GRU architecture but swaps BERT embeddings for higher‑dimensional Mistral sentence representations, testing sensitivity to underlying text encoder while keeping numerical feature fusion and gated temporal aggregation.

Core differences:

CNN vs recurrent: convolutional temporal pattern extraction vs sequential hidden state propagation.
RNN vs GRU vs LSTM: increasing gating sophistication (none, update/reset, full input/forget/output) for sequence representation.
GRU ablations: isolate impact of textual vs numerical/time metadata channels.
Transformer: self‑attention with positional embeddings and pooling replaces recurrence/convolution.
GRU Mistral: tests alternative (Mistral) text embedding space within the same gated sequence framework.
Baselines (in GRU notebook): non‑deep models over aggregated statistics provide reference points.