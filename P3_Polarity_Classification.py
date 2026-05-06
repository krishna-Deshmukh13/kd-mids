# ============================================================
# MI Practical 3 — Polarity-based Document Classification
# ============================================================
# Aim: Write a program to recognize whether a document is
#      positive or negative based on polarity words using
#      Naive Bayes classification.
# ============================================================
# Dataset: companyfeedback.csv
#          Columns: 'Comments' and 'Category'
# ============================================================
# POLARITY WORDS:
#   Positive: good, happy, excellent, awesome, amazing
#   Negative: bad, sad, terrible, awful, disappointing
# ============================================================

# ──────────────────────────────────────────────
# Cell [5–6] — Imports & Load Dataset
# ──────────────────────────────────────────────
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (columns: 'Comments' and 'Category')
data = pd.read_csv('companyfeedback.csv')

# ──────────────────────────────────────────────
# Cell [7, 10] — Define Polarity Words & Classify
# ──────────────────────────────────────────────

# Define polarity word lists
positive_words = ["good", "happy", "excellent", "awesome", "amazing"]
negative_words = ["bad",  "sad",   "terrible",  "awful",  "disappointing"]

# Function to classify based on polarity word count
def classify_document(document):
    pos_count = sum(1 for word in document.split()
                    if word in positive_words)
    neg_count = sum(1 for word in document.split()
                    if word in negative_words)
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'

# Apply to dataset
data['predicted_sentiment'] = data['Comments'].apply(classify_document)

print("Sample predictions:")
print(data[['Comments', 'predicted_sentiment']].head(10))

# ──────────────────────────────────────────────
# Cell [16–22] — Train Naive Bayes Model
# ──────────────────────────────────────────────

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    data['Comments'], data['Category'],
    test_size=0.2, random_state=42)

# Convert text to numerical features using CountVectorizer
# CountVectorizer = counts word frequency in each document
vectorizer     = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts  = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# Predict & Evaluate
y_pred   = classifier.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, zero_division=0)

print(f'Accuracy: {accuracy:.2f}')          # ~0.62
print('Classification Report:\n', report)

# ──────────────────────────────────────────────
# KEY CONCEPTS:
# Naive Bayes: Probabilistic classifier based on Bayes' theorem.
#              Assumes all features are independent.
#              Fast and works well for text.
# CountVectorizer: Counts word frequency per document.
# TF-IDF vs Count: TF-IDF penalizes very common words; Count doesn't.
# Expected accuracy: ~0.62
# NOTE: "Precision ill-defined" warning is normal for small datasets.
#       Use zero_division=0 in classification_report to suppress.
# ──────────────────────────────────────────────
