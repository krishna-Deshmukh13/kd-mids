# ============================================================
# MI Practical 2 — Text Classification using KNN
# ============================================================
# Aim: Text classification for sentimental analysis using
#      KNN algorithm on the IMDB dataset.
# ============================================================
# Dataset: IMDB Dataset.csv  (download from Kaggle.com)
# Place the CSV in the SAME folder as this script.
# ============================================================
# KNN ALGORITHM STEPS:
#   1. Select K (number of neighbors) — typically K=5
#   2. Calculate Euclidean distance from new point to all
#      training points
#   3. Take K nearest neighbors
#   4. Count data points in each category among K neighbors
#   5. Assign new point to category with maximum neighbors
# ============================================================

# ──────────────────────────────────────────────
# Cell [1] — Import Libraries
# ──────────────────────────────────────────────
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing

# ──────────────────────────────────────────────
# Cell [2] — Download NLTK Resources
# ──────────────────────────────────────────────
nltk.download('stopwords')      # Common words to ignore (the, a, is...)
nltk.download('punkt')          # Tokenizer
nltk.download('wordnet')        # For lemmatization
nltk.download('punkt_tab')      # Required in newer NLTK versions

# ──────────────────────────────────────────────
# Cell [3–4] — Load Dataset & Preprocess Text
# ──────────────────────────────────────────────
# Load IMDB dataset (columns: 'review' and 'sentiment')
df = pd.read_csv('IMDB Dataset.csv')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)                                # 1. Tokenize
    tokens = [w for w in tokens
               if w.lower() not in stop_words]                 # 2. Remove stopwords
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens] # 3. Lemmatize
    return ' '.join(tokens)

# Apply preprocessing to all reviews
df['review'] = df['review'].apply(preprocess_text)

# Encode labels: positive=1, negative=0
le = preprocessing.LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

# ──────────────────────────────────────────────
# Cell [5] — Split Data & Vectorize
# ──────────────────────────────────────────────
# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'],
    test_size=0.2, random_state=42)

# Convert text to numbers using TF-IDF
# TF-IDF = Term Frequency - Inverse Document Frequency
vectorizer    = TfidfVectorizer(max_features=5000)
X_train_bow   = vectorizer.fit_transform(X_train)
X_test_bow    = vectorizer.transform(X_test)

# Train KNN Classifier with K=5
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_bow, y_train)

# ──────────────────────────────────────────────
# Cell [6–7] — Evaluate & Visualize
# ──────────────────────────────────────────────
# Predict on test set
y_pred = clf.predict(X_test_bow)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)                    # ~0.45
print(classification_report(y_test, y_pred))

# Confusion Matrix visualization
conf_matrix = confusion_matrix(y_train, clf.predict(X_train_bow))
sns.heatmap(conf_matrix,
            annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix — KNN Sentiment')
plt.show()

# ──────────────────────────────────────────────
# KEY CONCEPTS:
# TF-IDF: Converts text to numbers — measures how important
#         a word is in a document relative to all documents.
# Lemmatization: "running" → "run" | "better" → "good"
# KNN accuracy on IMDB: ~0.45
# ──────────────────────────────────────────────
