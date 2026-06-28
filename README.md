# Titanic-Project
# Titanic Survival Prediction

**Intern ID:** CITS2712

---

## Project Overview

This project builds a machine learning pipeline to predict passenger survival on the Titanic using the classic Kaggle dataset. It covers end-to-end steps — data exploration, feature engineering, model training, and evaluation — comparing three classification algorithms to identify the best performer.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle / DataScienceDojo GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) |
| Rows | 891 |
| Columns | 12 |
| Target | `Survived` (0 = No, 1 = Yes) |

**Key Features Used:**
`Pclass` · `Sex` · `Age` · `Fare` · `Embarked` · `FamilySize` · `IsAlone` · `Title`

---

## Project Workflow

```
1. Load Data
2. EDA (distributions, survival patterns, correlations)
3. Feature Engineering (FamilySize, IsAlone, Title extraction)
4. Preprocessing (null handling, encoding, scaling)
5. Model Training (3 models)
6. Evaluation (Accuracy, ROC-AUC, CV Score, Confusion Matrix)
```

---

## Models Compared

| Model | Test Accuracy | ROC-AUC | 5-Fold CV |
|---|---|---|---|
| Logistic Regression | ~79% | ~0.85 | ~0.80 |
| Random Forest | ~82% | ~0.88 | ~0.83 |
| Gradient Boosting | ~81% | ~0.87 | ~0.82 |

> ✅ **Best Model: Random Forest**

---

## Feature Engineering

| New Feature | Description |
|---|---|
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if FamilySize == 1, else 0 |
| `Title` | Extracted from Name (Mr, Mrs, Miss, Rare etc.) |

---

## Key EDA Findings

- **Women survived at a much higher rate** than men (~74% vs ~19%)
- **1st class passengers** had the highest survival rate
- **Younger passengers** had a slightly higher chance of survival
- **Travelling alone** correlated with lower survival rate

---

## Output Files

| File | Description |
|---|---|
| `titanic_survival.py` | Main Python script |
| `survival_count.png` | Survival distribution bar chart |
| `survival_by_gender.png` | Survival split by gender |
| `survival_by_pclass.png` | Survival split by class |
| `age_distribution.png` | Age distribution by survival |
| `correlation_heatmap.png` | Feature correlation matrix |
| `confusion_matrix.png` | Random Forest confusion matrix |
| `roc_curve.png` | ROC-AUC curve |
| `feature_importance.png` | Random Forest feature importances |

---

## Tools & Libraries

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python titanic_survival.py
```

---

## Project Structure

```
├── titanic_survival.py     # Full ML pipeline
└── README.md               # Project documentation
```

---

*Intern ID: CITS2712*
