from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(accuracy_score,
                            classification_report,
                            precision_score,
                            recall_score,
                            f1_score,
                            roc_auc_score,
                            confusion_matrix)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'cgpa':          [8.5,7.2,9.1,6.5,8.8,7.0,9.5,6.8,
                      7.5,8.2,6.0,9.0,7.8,8.4,6.2],
    'projects':      [3,1,4,1,3,2,5,1,2,3,1,4,2,3,1],
    'internships':   [1,0,2,0,1,0,2,0,1,1,0,2,1,1,0],
    'communication': [8,6,9,5,8,6,9,5,7,8,5,9,7,8,5],
    'got_placed':    [1,0,1,0,1,0,1,0,1,1,0,1,1,1,0]
}
df = pd.DataFrame(data)

X = df[[
    'cgpa',
    'projects',
    'internships',
    'communication',]]
y = df['got_placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

print("\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not placed', 'Placed'],
            yticklabels=['Not placed', 'Placed'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()