# Virality Prediction - Evons
All notebooks tackle the same task: predict whether a news article will become “viral” (binary label derived by marking articles above the 95th percentile of Facebook engagements). All trainings use a 10 k-fold validation.

## Model variants:

MLP (`MLP.ipynb`): Concatenates title and caption embeddings (e.g. RoBERTa CLS vectors) and feeds them through a small feed-forward network for classification. This notebook calculates also performances of baseline classic ML models (Dummy, Logistic regression, random forest).

Source Embedding Model (`source_embedding_model.ipynb`): Replaces numeric engagement statistics with a learned embedding for the categorical media_source ID; reduces title and caption embeddings, embeds the source, concatenates all three, and classifies.

Average Engagement Model (`average_engagement_model.ipynb`): Extends the baseline by adding a per-source average historical engagement value (mean engagements per media_source) as a numeric feature, projecting it into the same latent space and concatenating with reduced text embeddings.

Gating Model (`gating_model.ipynb`): Uses the same average engagement feature but introduces a gating mechanism that learns, per feature dimension, how to blend transformed engagement information with the combined text representation.

Gating Model (Mistral) (`gating_model_mistral.ipynb`): Same gated fusion architecture as above, but operates on precomputed embeddings from a different underlying language model (Mistral instead of RoBERTa).

## Core differences:

What extra source signal is used (none, average engagement scalar, gating fusion, learned source embedding).
How auxiliary information is integrated (simple concat vs projected scalar vs gated blending vs categorical embedding).
Which underlying text embedding model is used (RoBERTa vs precomputed Mistral).
All remain intentionally simple classifiers over frozen (precomputed) text representations, differing only in how they incorporate source-level context.