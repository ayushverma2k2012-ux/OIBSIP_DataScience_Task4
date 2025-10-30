# Email Spam Detection — Machine Learning Project

## 📘 Objective
The objective of this project is to build and evaluate a **machine learning-based spam email classifier** that can automatically detect whether an email is *spam* or *non-spam (ham)*.  
Using Python and Natural Language Processing (NLP) techniques, the model analyzes email text data and learns patterns commonly found in spam messages such as phishing attempts, promotions, or scam content.

---

## 🧩 Steps Performed

### 1. *Data Loading*
- Imported the **Email Spam dataset** (from the provided source or CSV file).  
- Loaded the dataset into a pandas DataFrame for inspection.  
- Checked the data for missing or duplicate entries and verified label distribution between spam and non-spam emails.

### 2. *Exploratory Data Analysis (EDA)*
- Analyzed the frequency of spam vs ham messages.  
- Visualized the dataset using *count plots* and *word clouds* to identify the most common words in spam and ham emails.  
- Calculated message length statistics to understand text patterns across classes.

### 3. *Data Preprocessing*
- Cleaned and normalized email text by:
  - Removing punctuation, numbers, and stopwords.  
  - Converting all text to lowercase.  
  - Applying *tokenization* and *lemmatization* for text standardization.
- Converted text into numerical vectors using *TF-IDF Vectorizer* or *CountVectorizer*.  
- Split the dataset into **training (80%)** and **testing (20%)** sets.

### 4. *Model Training*
- Implemented multiple classification algorithms for spam detection:
  - *Multinomial Naive Bayes*  
  - *Logistic Regression*  
  - *Support Vector Machine (SVM)*  
- Trained models on vectorized text data and optimized performance using cross-validation.

### 5. *Model Evaluation*
- Evaluated each model using the following metrics:
  - *Accuracy*  
  - *Precision*  
  - *Recall*  
  - *F1 Score*  
- Displayed confusion matrices and classification reports for performance comparison.  
- Identified the best-performing model for classifying spam emails.

---

## ⚙ Tools & Libraries Used
| Category | Tools/Libraries |
|-----------|----------------|
| Data Handling | pandas, numpy |
| Text Processing | re, nltk, sklearn.feature_extraction.text (CountVectorizer, TfidfVectorizer) |
| Machine Learning | scikit-learn (MultinomialNB, LogisticRegression, SVC) |
| Evaluation | accuracy_score, confusion_matrix, classification_report |
| Visualization | matplotlib, seaborn, wordcloud |

---

## 🏁 Outcome
- Successfully built a **spam detection model** capable of classifying emails as spam or non-spam with high accuracy.  
- The *Multinomial Naive Bayes* and *SVM* models showed the most reliable performance for text classification tasks.  
- Gained practical experience in **Natural Language Processing (NLP)** and **text vectorization techniques**.  
- The project demonstrates an end-to-end ML workflow — from text preprocessing and feature extraction to model training and evaluation.

---
