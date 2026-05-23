# YouTube Trending Video Engagement Prediction

This project explores whether engagement levels on trending YouTube videos can be predicted using Natural Language Processing (NLP), sentence embeddings, and deep learning models. The project compares a transfer learning pipeline using Sentence Transformers and a neural network against a prompted Large Language Model baseline.

The goal was to classify YouTube videos into high engagement or low engagement categories using metadata such as titles, categories, clickbait scores, and tag counts.

---

## Dataset

Dataset used:
- Kaggle: `youtube-trending-videos-20202026`

The dataset contains trending YouTube videos along with metadata including:
- video title
- category
- engagement score
- clickbait score
- tag count
- capitalization usage
- subscriber statistics

The project removes duplicates and missing values, then converts engagement prediction into a binary classification problem using percentile-based thresholds.

---

## Features Used

- Video title
- Category
- Clickbait score
- Tag count
- Capitalization usage
- Engagement score

---

## Project Pipeline

### 1. Data Preprocessing
- Removed duplicates and null values
- Filtered invalid engagement scores
- Created binary labels for high vs low engagement
- Performed exploratory data analysis and visualizations
- Created train/validation/test splits

### 2. Sentence Embedding Model
Used:
- `sentence-transformers/all-MiniLM-L6-v2`

Text inputs combined:
- title
- category
- clickbait score
- tag count

The embeddings were passed into a PyTorch MLP classifier with:
- ReLU activations
- Dropout regularization
- BCEWithLogitsLoss
- Adam optimizer
- Early stopping

### 3. Large Language Model Baseline
Used:
- `Qwen/Qwen2.5-0.5B-Instruct`

The LLM was prompted to classify videos as:
- `0 = Low Engagement`
- `1 = High Engagement`

Predictions were compared against the embedding-based classifier.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn
- KaggleHub

---

## Results

### Sentence Transformer + MLP
- Accuracy: ~0.49
- Precision: ~0.49
- Recall: ~0.44
- F1 Score: ~0.46

### Qwen LLM Baseline
- Accuracy: ~0.49
- Precision: ~0.49
- Recall: ~0.95
- F1 Score: ~0.65

The experiments showed that predicting engagement using limited metadata is challenging because many features had weak correlations with engagement scores. The LLM baseline achieved higher recall but tended to overpredict high-engagement videos.
