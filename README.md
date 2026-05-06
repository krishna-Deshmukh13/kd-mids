# MI for Data Science — Machine Intelligence
## TE-AIML | Alard College of Engineering & Management | 2023–24

---

## 📁 Practicals

| File | Practical | Key Concept |
|------|-----------|-------------|
| `P1_Titanic_Preprocessing.py` | Titanic dataset pre-processing | Missing values, Label Encoding |
| `P2_KNN_Sentiment_Analysis.py` | KNN text classification (IMDB) | TF-IDF, KNN, NLP Pipeline |
| `P3_Polarity_Classification.py` | Polarity document classification | Naive Bayes, CountVectorizer |
| `P4_KMeans_Clustering.py` | K-Means clustering (8 points) | Centroids, Manhattan distance |

---

## 📂 Required Datasets

| Dataset | Download From | Used In |
|---------|--------------|---------|
| `titanic.csv` | kaggle.com/datasets/titanic | P1 |
| `IMDB Dataset.csv` | kaggle.com/datasets/imdb-dataset-of-50k-movie-reviews | P2 |
| `companyfeedback.csv` | Create manually or download | P3 |

> **Place all datasets in the SAME folder as the `.py` files.**

---

## ▶️ How to Run

```bash
python P1_Titanic_Preprocessing.py
python P2_KNN_Sentiment_Analysis.py
python P3_Polarity_Classification.py
python P4_KMeans_Clustering.py
```

Or open in **Jupyter Notebook** — copy each cell into a cell and run with `Shift+Enter`.

---

## 📦 Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

---

## 🔑 Key Numbers to Remember

| Practical | Key Values |
|-----------|-----------|
| P1 Titanic | Total:891, Survived:342, Male survival:18.9%, Female:74.2% |
| P2 KNN | IMDB accuracy: ~0.45, K=5, max_features=5000 |
| P3 Naive Bayes | accuracy: ~0.62 |
| P4 K-Means | M1=[0.1225,0.765], M2=[0.2475,0.275], SSE=0.045, 2 iterations |

---

## 📊 P4 K-Means Exam Answers

- **P6 belongs to:** Cluster 2 (C2)
- **C2 population:** P5, P6, P7, P8 (4 points)
- **Updated M1:** [0.1225, 0.765]
- **Updated M2:** [0.2475, 0.275]
