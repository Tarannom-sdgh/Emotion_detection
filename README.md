# Emotion Classification Repository

This repository contains code for emotion classification using various models including BiLSTM, BERT fine-tuning, and NB-SVM.

## Data

The data is cleaned and extracted from 3 datasets. Data was merged and cleaned and labels that had poor performance were removed.
The data that had been used:
https://www.kaggle.com/datasets/debarshichanda/goemotions
or https://huggingface.co/datasets/go_emotions
https://www.kaggle.com/datasets/parulpandey/emotion-dataset
https://www.kaggle.com/datasets/dtughdr/dataset-3

## 1. BiLSTM Model

This code implements a Bidirectional Long Short-Term Memory (BiLSTM) model for emotion classification. The model is trained on two datasets for single-label emotion detection.
The emotions considered are sadness, joy, love, anger, fear, and surprise. The datasets have been extracted from Kaggle and achieved high scores.

### Results

- Matthews Correlation Coefficient: 0.9260
- Macro F1 score: 0.91
- Precision per class: [0.96809422 0.94461538 0.89452167 0.96721311 0.89673913 0.91942149]
- Recall per class: [0.96892413 0.97333085 0.83895706 0.92913386 0.937016 0.77797203] (sadness, joy, love, anger, fear, and surprise)
- Training Accuracy: 0.932

### Usage

To use the model, utilize the `model.predict` method. The dataset is also provided for cloning.

## 2. Fine-tuned BERT Model

This code fine-tunes a DistilBERT model from the Transformers library on an Emotion Database obtained from Kaggle.

### Validation Results

| Epoch | Training Loss | Validation Loss | Accuracy | F1       |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.134800      | 0.158145        | 0.934500 | 0.935167 |
| 2     | 0.087000      | 0.157977        | 0.937000 | 0.937520 |
| 3     | 0.076100      | 0.153011        | 0.939500 | 0.939542 |

## 3. Multi-Label Classification

This code implements a Naive Bayes model for multi-label classification. The dataset used is a combination of cleaned datasets from previous sections and a Reddit dataset, allowing for multiple labels per row. The data underwent thorough cleaning to remove inaccurate labels and merge datasets.
The emotions considered are anger, disappointment, disgust, fear, joy, love, nervousness, surprise, neutral, sadness, amusement and excitement

### Usage

To use the model please configure config.yml based on your usage. Set train: false to use the model. Set train: true to train your own model based on your own data, please update train_data_path to directory that your data is stored in config.yml.
The file config.yml is to address your data and model.

### Results

On all-data:

- ROC AUC: 85%
- Hamming Loss: 0.084
- F1 Score (micro): 0.60 (macro): 0.47 -with a threshold of > 0.33 for considering the result as 1 if the threshold is not satisfied values that are higher than sum of variance and mean are considered 1.

On data with removed near-duplicates and outliers:

- ROC AUC: 80%
- F1 Score (micro): 0.50 (macro): 0.37 -with a threshold of > 0.15 for considering the result as 1 if the threshold is not satisfied values that are higher than sum of variance and mean are considered 1.
