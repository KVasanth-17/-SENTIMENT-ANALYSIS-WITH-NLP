
# Sentiment Analysis using NLP

*COMPANY*: CODTECH IT SOLUTIONS  

*NAME*: VASANTH KANDIBANDA  

*INTERN ID*: CT08DZ1238 

*DOMAIN*: MACHINE LEARNING  

*DURATION*: 8 WEEKS  

*MENTOR*: NEELA SANTOSH  

*DESCRIPTION OF THE TASK*: 

This project performs **Sentiment Analysis** on customer reviews using **TF-IDF vectorization** and **Logistic Regression**, two widely used approaches for text classification. The workflow includes preprocessing, model training, evaluation, visualization, and saving the trained model.

---

## 📘 Description of the Code

### 1. Importing Libraries
The script begins by importing required libraries:
- **pandas, numpy** for data handling.
- **scikit-learn** for machine learning tasks (TF-IDF, Logistic Regression, evaluation metrics).
- **matplotlib, seaborn** for visualizations.
- **nltk** for text preprocessing.
- **datasets** from Hugging Face to load the IMDb dataset.
- **joblib** for saving the trained model.

---

### 2. Loading Dataset
We use the **IMDb movie reviews dataset**, a benchmark dataset for sentiment classification containing 50,000 labeled reviews. The Hugging Face `datasets` library is used to load the dataset and convert it to a Pandas DataFrame.

---

### 3. Preprocessing
A custom function `clean_text()` is defined to:
- Lowercase the text.
- Remove HTML tags and line breaks.
- Eliminate special characters, keeping only alphanumeric text.
- Normalize whitespace.

The cleaned reviews are stored in a new DataFrame column (`clean_text`).

---

### 4. Train-Test Split
The dataset is split into **80% training** and **20% testing** using `train_test_split`. Stratification ensures both subsets maintain the same class distribution.

---

### 5. TF-IDF + Logistic Regression Pipeline
The pipeline consists of:
1. **TF-IDF Vectorizer**: Converts text into numerical features, considering both unigrams and bigrams (`ngram_range=(1,2)`).
2. **Logistic Regression**: A linear classifier effective for binary sentiment classification.

Hyperparameter tuning is performed with **GridSearchCV**. Parameters such as document frequency thresholds (`min_df`, `max_df`), n-gram ranges, and regularization strength (`C`) are optimized.

---

### 6. Model Training and Evaluation
The model is trained on the training set and evaluated on the test set. Metrics include:
- **Accuracy**: Overall correctness.
- **Precision, Recall, F1-Score**: For balanced evaluation of both classes.
- **ROC AUC**: Ability to distinguish positive vs negative classes.
- **Classification Report**: Detailed per-class metrics.
- **Confusion Matrix**: Visualized with Seaborn heatmap.

Additionally, an **ROC Curve** is plotted to show the trade-off between true positive rate and false positive rate.

---

### 7. Model Saving
The trained pipeline (TF-IDF + Logistic Regression) is saved as a `.joblib` file. This allows easy reuse without retraining.

---

### 8. Example Predictions
Sample reviews are tested with the model, and predictions are displayed as **positive** or **negative**, along with probability scores.

---

## 📌 Conclusion
This project demonstrates the complete workflow of a **sentiment analysis system**: preprocessing raw text, training and evaluating a model, visualizing results, and saving the trained pipeline. Using **TF-IDF** for feature extraction and **Logistic Regression** for classification provides a strong and interpretable baseline for text sentiment classification tasks.

---

*OUTPUT*: 

Model Performance:
Accuracy : 0.9202
Precision: 0.9121
Recall   : 0.9300
F1 Score : 0.9210
ROC AUC  : 0.9747

<img width="522" height="470" alt="Image" src="https://github.com/user-attachments/assets/304ffeb2-55c4-4e15-af44-2363c8b1c6de" />

<img width="536" height="470" alt="Image" src="https://github.com/user-attachments/assets/a82f219a-7a67-458d-9e1d-587de0919581" />

TEXT: What a fantastic movie! The story and acting were superb.
Prediction: Positive | Prob_Positive: 0.9455

TEXT: Terrible film. I walked out after 20 minutes.
Prediction: Negative | Prob_Positive: 0.0023

TEXT: Mediocre, some good moments but overall forgettable.
Prediction: Negative | Prob_Positive: 0.0074

