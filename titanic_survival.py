# =============================================================
# Titanic Survival Prediction
# Intern ID: CITS2712
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')

# =============================================================
# 1. LOAD DATA
# =============================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nNull Values:")
print(df.isnull().sum())

# =============================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================

# Survival count
plt.figure(figsize=(6, 4))
df['Survived'].value_counts().plot(kind='bar', color=['salmon', 'steelblue'], edgecolor='black')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'], rotation=0)
plt.title('Survival Count')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('survival_count.png', dpi=150)
plt.show()

# Survival by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Gender')
plt.tight_layout()
plt.savefig('survival_by_gender.png', dpi=150)
plt.show()

# Survival by passenger class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Passenger Class')
plt.tight_layout()
plt.savefig('survival_by_pclass.png', dpi=150)
plt.show()

# Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, palette='Set2', bins=30)
plt.title('Age Distribution by Survival')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr(),
            annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()

# =============================================================
# 3. FEATURE ENGINEERING
# =============================================================

df_model = df.copy()

# Fill missing values
df_model['Age'].fillna(df_model['Age'].median(), inplace=True)
df_model['Embarked'].fillna(df_model['Embarked'].mode()[0], inplace=True)
df_model.drop(columns=['Cabin'], inplace=True)  # too many nulls

# Encode categorical
le = LabelEncoder()
df_model['Sex'] = le.fit_transform(df_model['Sex'])           # male=1, female=0
df_model['Embarked'] = le.fit_transform(df_model['Embarked']) # C=0, Q=1, S=2

# New features
df_model['FamilySize'] = df_model['SibSp'] + df_model['Parch'] + 1
df_model['IsAlone'] = (df_model['FamilySize'] == 1).astype(int)
df_model['Title'] = df_model['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df_model['Title'] = df_model['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df_model['Title'] = df_model['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
df_model['Title'] = le.fit_transform(df_model['Title'])

# Final feature set
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title']
X = df_model[features]
y = df_model['Survived']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# =============================================================
# 4. MODEL TRAINING
# =============================================================

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 4),
        'CV Score':  round(cv_score, 4)
    }

results_df = pd.DataFrame(results).T
print("\n=== Model Comparison ===")
print(results_df)

# =============================================================
# 5. BEST MODEL — RANDOM FOREST (detailed evaluation)
# =============================================================

best_model = models['Random Forest']
y_pred_best = best_model.predict(X_test)

print("\n=== Random Forest — Classification Report ===")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_best),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived','Survived'],
            yticklabels=['Not Survived','Survived'])
plt.title('Confusion Matrix — Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:,1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue', label=f"AUC = {results['Random Forest']['ROC-AUC']}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.title('ROC Curve — Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

# Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(7, 5))
feat_imp.plot(kind='barh', color='mediumpurple', edgecolor='black')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# =============================================================
# 6. SUMMARY
# =============================================================

print("\n=== FINAL SUMMARY ===")
print(f"Best Model      : Random Forest")
print(f"Test Accuracy   : {results['Random Forest']['Accuracy']*100:.2f}%")
print(f"ROC-AUC Score   : {results['Random Forest']['ROC-AUC']}")
print(f"5-Fold CV Score : {results['Random Forest']['CV Score']*100:.2f}%")
print("Intern ID       : CITS2712")
