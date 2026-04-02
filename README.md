# AI-Generated Text Detection via Transformers

## Overview
A research-focused project aimed at addressing academic integrity concerns by detecting ChatGPT-generated text across diverse and complex domains (Finance, Medical, Wikipedia). The study involved building a Bi-directional LSTM baseline from scratch and comparing it against a fine-tuned Transformer (BERT) model. Explainable AI (XAI) techniques were also applied to interpret the linguistic patterns distinguishing AI from human text.

## Tech Stack
* **Language:** Python
* **Deep Learning Frameworks:** PyTorch, Hugging Face Transformers
* **Models:** Bi-LSTM, BERT
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)

## Key Features & Methodology

### 1. Multi-Domain Text Classification
* Compiled and preprocessed a balanced dataset containing both human-written and AI-generated texts across specialized domains: Finance, Medicine, and General Knowledge (Wikipedia).
* Addressed class imbalance and performed extensive text tokenization and embedding.

### 2. Model Implementation & Fine-Tuning
* **Baseline Model (Bi-LSTM):** Implemented a Bi-directional LSTM model from scratch using PyTorch to capture sequential dependencies in the text. Achieved a baseline accuracy of ~90%.
* **Advanced Model (Fine-tuned BERT):** Fine-tuned a pre-trained BERT model on the multi-domain dataset to leverage deep bidirectional representations.
* **Performance:** The BERT model significantly outperformed the baseline, achieving **94.1% Accuracy** and an impressive **0.98 ROC-AUC** score.

### 3. Interpretability via SHAP (Explainable AI)
* To understand *why* the model made specific classifications, SHAP values were calculated for the input features.
* **Findings:** The SHAP analysis revealed that AI-generated texts tend to exhibit higher structural formality and specific repetitive linguistic patterns compared to human-authored texts.

## Architecture (Methodology)
* `data_loader` : Scripts for importing and tokenizing the multi-domain text datasets.
* `models/bilstm` : PyTorch implementation of the baseline Bi-LSTM architecture.
* `models/bert_finetune` : Training loop and fine-tuning configurations for the BERT model.
* `evaluation_shap` : Inference scripts outputting Accuracy, ROC-AUC, and SHAP value visualizations.
