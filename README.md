# RAER - Retrieval-Augmented EEG Reasoning for Explainable Seizure Detection

**Self-Supervised Learning · EEG Signal Processing · PyTorch · FAISS · LLMs**

RAER is a research prototype for **EEG-based seizure classification and explainability**. It combines contrastive representation learning, a lightweight classifier, similarity-based retrieval, and LLM-generated explanations to provide context alongside model predictions.

## Overview

EEG seizure detection is challenging because seizure events are rare and physiological recordings vary substantially. RAER learns representations from unlabeled EEG segments, classifies seizure versus non-seizure windows, and retrieves similar historical windows to contextualize predictions.

**Pipeline:** EEG recordings → filtering and windowing → SimCLR-style 1D CNN → 128-dimensional embeddings → MLP seizure classifier → FAISS nearest-neighbor retrieval → LLM-generated explanations.

## Dataset and preprocessing

The project uses a **nine-patient subset** of the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).

- EEG bandpass filtering (**0.5–40 Hz**) and per-channel z-score normalization
- Segmentation into **10-second windows** with overlapping segments
- Seizure labels derived from annotated seizure intervals
- **95,501 windows**, including **95 seizure windows**, in the reported experiment

## Models and methods

**Self-supervised representation learning:** A SimCLR-style **1D CNN encoder** learns 128-dimensional EEG embeddings using **InfoNCE contrastive loss** and augmentations such as Gaussian noise, scaling, and time shifts.

**Classification:** A lightweight **MLP classifier** is trained on frozen EEG embeddings to predict seizure versus non-seizure windows.

**Retrieval and explanations:** **FAISS** indexes normalized embeddings for cosine-similarity search and retrieves the **five nearest windows**. Retrieved examples inform **LLM-generated, evidence-grounded explanations** and help highlight cases where classification and retrieved evidence disagree.

## Visualizations

### EEG embedding space (t-SNE)

![t-SNE visualization of learned EEG embeddings](images/eeg_embeddings_tsne.png)

*A two-dimensional view of sampled learned EEG representations.*

### Classifier training

![Classifier training and validation curves](images/classifier_training.png)

*Training diagnostics for the downstream seizure classifier.*

### Contrastive pretraining

![Contrastive learning loss over training epochs](images/contrastive_pretraining_loss.png)

*InfoNCE loss during self-supervised encoder training.*

## Evaluation

The final project report describes a held-out **window-level validation** experiment with approximately **0.970 ROC-AUC** and **0.556 F1-score**. Because seizure windows represent only about **0.1%** of the dataset, accuracy alone is not an informative measure of seizure detection quality.

**Evaluation caveat:** The report mentions patient-wise evaluation in some sections but describes an **80/20 stratified split** in the methods. These results should **not** be presented as verified performance on entirely unseen patients until the exact splitting protocol is confirmed. Overlapping windows also require careful leakage checks.

## Technology stack

**Python · PyTorch · MNE · NumPy · scikit-learn · FAISS · Matplotlib · t-SNE · LLMs**

## Repository

The accompanying notebook contains the research workflow and experimental visualizations. The `images/` directory contains figures exported from notebook outputs for README display.


