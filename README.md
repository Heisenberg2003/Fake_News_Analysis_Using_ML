# Fake News Detection using NLP and Machine Learning

This project focuses on detecting **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. With the rising spread of misinformation online, this tool aims to classify news articles as either **Real** or **Fake** based on their content.

---

## Tech Stack & Libraries
- **Language:** Python
- **Libraries:** 
  - `pandas`, `numpy`
  - `scikit-learn`
  - `nltk` (Natural Language Toolkit)
  - `matplotlib`, `seaborn` (for visualizations)
- **NLP Techniques:** 
  - Tokenization, Stopword Removal, Stemming
  - TF-IDF Vectorization
- **Models Used:** 
  - Logistic Regression
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)

---

## Dataset
- Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- The dataset includes two CSV files:
  - `Fake.csv`: Contains fake news articles
  - `True.csv`: Contains real news articles

---

## Project Workflow

1. **Data Collection**
   - Combined `Fake.csv` and `True.csv` into a single dataset.
   - Added labels: `0` for fake, `1` for real.

2. **Text Preprocessing**
   - Removed punctuation, symbols, and stopwords.
   - Converted text to lowercase.
   - Applied stemming using NLTK's PorterStemmer.

3. **Vectorization**
   - Transformed text into numerical form using **TF-IDF Vectorizer**.
   - Optionally compared with CountVectorizer (BoW model).

4. **Model Training & Evaluation**
   - Trained models: Logistic Regression, Naive Bayes, SVM
   - Evaluation metrics: Accuracy, Precision, Recall, F1 Score
   - Visualized performance using Confusion Matrix and plots.

5. *(Optional)* **Deployment**
   - Ready for deployment via Flask or Streamlit (not included in this repo by default).

---

## Results
| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | ~96%     |
| Naive Bayes         | ~93%     |
| SVM                 | ~95%     |

> Logistic Regression provided the best balance between speed and accuracy.

---

## Features
- Detects whether a news article is fake or real
- Clean and reproducible machine learning pipeline
- Modular codebase for easy updates or deployment
- Preprocessing pipeline included using NLTK

---

## Skills Demonstrated
- Real-world implementation of **NLP**
- Application of **Supervised Machine Learning**
- Data Cleaning & Feature Engineering
- Model Evaluation and Selection
- Visualization of Model Performance
