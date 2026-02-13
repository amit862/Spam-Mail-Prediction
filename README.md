# 📧 Spam Mail Prediction using Logistic Regression

This project implements a **Spam Mail Classifier** using **Logistic Regression** and **Natural Language Processing (NLP)** techniques to automatically classify emails as **Spam** or **Ham (Not Spam)**.

---

## 🚀 Project Overview

Email spam detection is a classic text classification problem.  
In this project, machine learning technique are applied to analyze email content and predict whether a given message is spam or legitimate.

The model is trained using **Logistic Regression** with **TF-IDF vectorization**, ensuring good performance and generalization on unseen data.

---

## 🧠 Machine Learning Approach

- **Algorithm:** Logistic Regression  
- **Text Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)  
- **NLP Techniques:**  
  - Text cleaning  
  - Stopword removal  
  - Tokenization  
  - N-grams  

---

## 📊 Dataset

- Labeled email dataset containing:
  - `spam`
  - `ham` (not spam)
- Dataset is split into **training** and **testing** sets
- Class imbalance handled using appropriate techniques

---

## ⚙️ Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib / Seaborn (for evaluation)  

---

## 🔁 Workflow

1. Data Loading  
2. Text Preprocessing  
3. Feature Extraction using TF-IDF  
4. Model Training (Logistic Regression)  
5. Model Evaluation  
6. Prediction on New Emails  

---

## 📈 Model Evaluation Metrics

The model is evaluated using multiple performance metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

These metrics help ensure the model performs well, especially on spam detection.

---

## 🧪 Sample Prediction

```python
email = ["Congratulations! You have won a free lottery ticket."]
prediction = model.predict(vectorizer.transform(email))
print(prediction)
