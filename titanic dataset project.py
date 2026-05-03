from unicodedata import numeric

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix)

# load directly from url — no download needed
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
df.columns = df.columns.str.title().str.lower().str.replace(' ','_')
print(df.columns)
print(df.isnull().sum())
print(df.duplicated().sum())
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop(columns=['cabin'], inplace=True)
df.drop(columns=['passengerid','name','ticket'], inplace=True)
#encoding categorial variables
df['sex'] = df['sex'].map({'female':0 , 'male': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

print(df.isnull().sum())
print(df.shape)
print(df.info())
print("Overall Survival Rate")
print(df['survived'].value_counts(normalize=True) * 100)
print("\nSurvival Rate by Gender")
print(df.groupby('sex')['survived'].mean() *100)
print("\nSurvival Rate by Passenger class")
print(df.groupby('pclass')['survived'].mean() *100)
df['Age Group'] = pd.cut(
    df['age'],
    bins=[0, 12, 18,35, 60, 100],
    labels=['child', 'teen', 'young adult', 'adult', 'senior']
)
print("\nSurvival Rate by Age group")
print(df.groupby('Age Group')['survived'].mean() * 100)

#visualisation
print(df.select_dtypes(include='number').columns)

print(df.groupby('pclass')['survived'].agg(['mean', 'max', 'min', 'count']))
print(df.groupby('fare').agg(
    mean_fare=('fare', 'mean'),
    max_fare=('fare', 'max'),
    min_fare=('fare', 'min'),
    avg_age=('age', 'mean')
))

print(df.select_dtypes(include='number').corr())
#positive correlation greatest for sibsp, parch  (0.41)
#negative correlation (-0.54) sex, survived -means no.of female(0) survived more or no. of male survived less

#heatmap
plt.figure(figsize=(9,5))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1, vmax=1,
    linewidth=0.5
)
plt.title('Correlation Heatmap')
plt.tight_layout()
#plt.show()

plt.figure(figsize=(9,5))
sns.scatterplot(data=df.corr(numeric_only=True), x ='survived', y = 'age')
sns.regplot(
    data=df.corr(numeric_only=True),
    x = 'survived',
    y = 'age'
)
plt.title('Age vs Survived')
plt.tight_layout()
#plt.show()


fig, axes = plt.subplots(2,2, figsize=(12,10))


sns.countplot(data=df, x='survived', color='steelblue',ax = axes[0,0])
axes[0,0].set_title('Survival Rate by Age')
axes[0,0].set_xticklabels(['Not Survived', 'Survived'])

sns.barplot(data=df, x='sex', y='survived', color='pink', ax = axes[0,1])
axes[0,1].set_title('Survival Rate by Gender')
axes[0,1].set_xticklabels(['Female', 'Male'])

sns.barplot(data=df, x='pclass', y='survived', color='mediumseagreen', ax = axes[1,0])
axes[1,0].set_title('Survival Rate by Pclass')
axes[1,0].set_xticklabels(['1', '2', '3'])

sns.histplot(data=df, x='age', hue='survived', bins=20,  palette='coolwarm', ax = axes[1,1])
axes[1,1].set_title('Survival Rate by Age')
plt.tight_layout()
#plt.show()

plt.figure(figsize=(9,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidth=0.5
            )
plt.title('Correlation Heatmap')
plt.tight_layout()
#plt.show()

#ML
df.drop(columns=['Age Group'], inplace=True)
X = df.drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=('Not Survived', 'Survived'),
            yticklabels=('Not Survived', 'Survived'))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
#plt.show()

feat_imp = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='coolwarm')
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
print(feat_imp)