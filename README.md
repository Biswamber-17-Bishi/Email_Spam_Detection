# Email Spam Detection Using NLP

Classify SMS/email messages as **Spam** or **Legitimate (Ham)** using NLP and machine learning.

## Project Structure
See master prompt structure. Key modules in `src/`.

## Setup
```bash
pip install -r requirements.txt
```

## Run Pipeline
```bash
python src/download_dataset.py
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
```

## Run App
```bash
streamlit run app.py
```

## Models
Multinomial Naive Bayes, Logistic Regression, Linear SVM (best F1 ~0.92).

## Dataset
UCI SMS Spam Collection, downloaded programmatically into `data/raw/`.