from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data = {
    'cgpa':          [8.5, 7.2, 9.1, 6.5, 8.8, 7.0, 9.5, 6.8,
                      7.5, 8.2, 6.0, 9.0, 7.8, 8.4, 6.2],
    'projects':      [3, 1, 4, 1, 3, 2, 5, 1, 2, 3, 1, 4, 2, 3, 1],
    'internships':   [1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 2, 1, 1, 0],
    'communication': [8, 6, 9, 5, 8, 6, 9, 5, 7, 8, 5, 9, 7, 8, 5],
    'got_placed':    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)

X = df[['cgpa', 'projects', 'internships', 'communication']]
y = df['got_placed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

#logisticregression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#linearregression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("r2:", r2_score(y_test, y_pred))

#decisiontree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.tree import export_text
print(export_text(model, feature_names = list(X.columns)))

#Randomtreeclassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
feat_imp = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(feat_imp)