# ============================================================
# MI Practical 1 — Titanic Dataset Data Pre-processing
# ============================================================
# Aim: Access the open-source Titanic dataset and apply data
#      pre-processing techniques on the raw dataset.
# ============================================================
# Dataset: titanic.csv  (download from Kaggle.com)
# Place titanic.csv in the SAME folder as this script.
# ============================================================
# 6 STEPS OF DATA PRE-PROCESSING:
#   1. Import Libraries
#   2. Import / Load the Data
#   3. Check for Missing Values
#   4. Arrange / Encode Categorical Data
#   5. Do Scaling
#   6. Split into Train / Validation / Test Sets
# ============================================================

# ──────────────────────────────────────────────
# Cell [1] — Import Libraries
# ──────────────────────────────────────────────
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# Cell [2–4] — Load & View Dataset
# ──────────────────────────────────────────────
# Load the CSV file into a DataFrame
data = pd.read_csv('titanic.csv')

# View first 5 rows
print(data.head())

# Check data types of each column
print(data.dtypes)

# ──────────────────────────────────────────────
# Cell [5–9] — Explore the Data
# ──────────────────────────────────────────────
# Shape: (rows, columns)
print("Shape:", data.shape)         # Output: (891, 12)

# Detailed info about columns & non-null counts
data.info()

# Survival counts
print('Total passengers:', len(data))                               # 891
print('Survived:', len(data[data['Survived'] == 1]))                # 342
print('Did not survive:', len(data[data['Survived'] == 0]))         # 549

# Gender counts
print(data['Sex'].value_counts())   # male:577, female:314

# Survival % by gender
print('% male survived:   ', 100 * np.mean(data['Survived'][data['Sex'] == 'male']))
print('% female survived: ', 100 * np.mean(data['Survived'][data['Sex'] == 'female']))

# ──────────────────────────────────────────────
# Cell [10–14] — Handle Missing Values
# ──────────────────────────────────────────────
# Make a copy — NEVER modify original data directly
df2 = data.copy()

# Check all missing values
print(df2.isnull().sum())
# Age: 177 missing | Cabin: 687 missing | Embarked: 2 missing

# Fix Age: fill with mean age (29)
print("Mean Age:", int(data['Age'].mean()))     # 29
df2['Age'] = df2['Age'].fillna(np.mean(df2['Age']))

# Fix Embarked: fill with mode (most common = 'S')
print("Mode Embarked:", df2['Embarked'].mode()[0])   # S
df2['Embarked'].fillna(df2['Embarked'].mode()[0], inplace=True)

# Fix Cabin: fill with mode (most common cabin)
df2['Cabin'].fillna(df2['Cabin'].mode()[0], inplace=True)

# Verify — should show 0 for all
print("After fix:\n", df2.isnull().sum())

# ──────────────────────────────────────────────
# Cell [15] — Encode Categorical Data
# ──────────────────────────────────────────────
# Convert Sex column: male=1, female=0
df2['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Check correlation
print("Correlation PassengerId vs Pclass:", df2['PassengerId'].corr(df2['Pclass']))

# ──────────────────────────────────────────────
# KEY OUTPUT TO REMEMBER:
# Total: 891 | Survived: 342 | Did not: 549
# Male survival: ~18.9%  |  Female survival: ~74.2%
# 1st class: 62.9%  |  2nd: 47.3%  |  3rd: 24.2%
# Missing — Age:177, Cabin:687, Embarked:2
# ──────────────────────────────────────────────
