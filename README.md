# Sentiment Analysis: NLP Machine Learning Pipeline

This project develops a complete **Natural Language Processing (NLP) pipeline** for performing **sentiment analysis** on movie reviews. It demonstrates both **rule-based** methods and **machine learning models** to classify sentiments as **positive (1)**, **neutral (0)**, or **negative (-1)**.

---

##  Contents

- **[Project Goals](#project-goals)**
- **[Tools & Libraries](#tools--libraries)**
- **[Data Description](#data-description)**
- **[Approach](#approach)**
  - **[Heuristic-Based Classification](#heuristic-based-classification)**
  - **[ML-Based Sentiment Detection](#ml-based-sentiment-detection)**
  - **[Feature Engineering: BoW & TF-IDF](#feature-engineering-bow--tf-idf)**
- **[Modeling & Tuning](#modeling--tuning)**
- **[Performance Review](#performance-review)**
- **[Running the Project](#running-the-project)**

---

##  Project Goals

- Build a baseline sentiment classifier for textual data.
- Compare rule-based logic with trained machine learning models.
- Convert raw reviews into machine-readable formats.
- Explore vectorization techniques like **Bag-of-Words** and **TF-IDF**.
- Apply **Grid Search** for parameter tuning.
- Evaluate final models using unseen test data.

---

##  Tools & Libraries

- **Python 3**
- **pandas**, **numpy**
- **scikit-learn**
- **nltk** (for text preprocessing)
- **Jupyter Notebook**

---


##  Data Description

The dataset contains text reviews along with sentiment labels. It's divided as follows:

- `train.csv` – for model training  
- `validation.csv` – for hyperparameter tuning  
- `test.csv` – for final performance evaluation  

Each entry includes:
- **Review text**
- **Sentiment label** (1 = Positive, 0 = Neutral, -1 = Negative)

---

##  Approach

###  Heuristic-Based Classification

We begin with a basic logic using predefined sets of **positive** and **negative** words. The sentiment is assigned based on which type appears more frequently:


```python
if positives > negatives:
    sentiment = 1
elif negatives > positives:
    sentiment = -1
else:
    sentiment = 0
```
## ML-Based Sentiment Detection

We then train a Logistic Regression model to predict sentiments based on vectorized features of the text reviews.

## Feature Engineering: BoW & TF-IDF
To convert text into numerical features, we use:

  - **Bag-of-Words (BoW) – Captures word frequency**

  - **TF-IDF (Term Frequency–Inverse Document Frequency) – Weighs words based on relevance**

Implemented using CountVectorizer and TfidfVectorizer from scikit-learn.

## Modeling & Tuning
  - **Applied Logistic Regression**

  - **Used GridSearchCV to tune:**

    - **Regularization type: 'l1', 'l2'**

    - **Regularization strength: 'C' values over a range**

  - **Leveraged PredefinedSplit to manage training and validation sets during cross-validation.**

## Performance Review
  - **Evaluated accuracy on both validation and test datasets.**

  - **Analyzed model coefficients to identify the most impactful words.**

  - **Compared results between:**

     - **Rule-based method**

     - **BoW + Logistic Regression**

     - **TF-IDF + Logistic Regression**

# Running the Project
Install dependencies:

```
pip install -r requirements.txt
```
Run the notebook:
Open sentiment_analysis_pipeline.ipynb in Jupyter and execute cells sequentially.

