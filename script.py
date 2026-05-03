from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data = {
    'hours_studied': [1,2,3,4,5,6,7,8,9,10,
                      2,4,6,8,3,5,7,9,1,6],
    'sleep_hours':   [5,6,7,6,8,7,8,9,6,8,
                      5,7,6,8,6,7,8,9,5,7],
    'absences':      [8,6,5,4,3,2,2,1,7,1,
                      7,4,3,2,6,3,2,1,8,2],
    'passed':        [0,0,0,1,1,1,1,1,0,1,
                      0,1,1,1,0,1,1,1,0,1]
}
df = pd.DataFrame(data)

X = df[['hours_studied', 'sleep_hours', 'absences']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

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

print("MAE:", mean_absolute_error(y_test,y_pred))
print("R2:", r2_score(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.tree import export_text
print(export_text(model, feature_names=list(X.columns)))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
feat_imp = pd.Series(
    model.feature_importances_,
    index = X.columns
).sort_values(ascending=False)
print(feat_imp)

