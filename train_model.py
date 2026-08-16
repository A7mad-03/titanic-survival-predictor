"""
Train a Titanic survival-prediction model and save it for the Streamlit app.
Mirrors the course workflow: clean -> encode -> split -> train -> evaluate -> save.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("/mnt/user-data/uploads/titanic.csv")

# ---- Data Engineering (mirrors Data_Eng_1 slides: fill missing, encode) ----
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df["Sex_enc"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked_enc"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

features = ["Pclass", "Sex_enc", "Age", "SibSp", "Parch", "Fare", "Embarked_enc"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate (mirrors ML_3 slides: cross-validation + accuracy) ----
test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
cv_scores = cross_val_score(model, X, y, cv=5)
cm = confusion_matrix(y_test, test_preds)

print(f"Test accuracy: {test_acc:.3f}")
print(f"Cross-val mean: {cv_scores.mean():.3f}  (std: {cv_scores.std():.3f})")
print("Confusion matrix:\n", cm)

# feature importance, for a nice "what drives the prediction" chart in the app
importances = dict(zip(features, model.feature_importances_))
print("Feature importances:", importances)

joblib.dump({
    "model": model,
    "features": features,
    "test_accuracy": float(test_acc),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "confusion_matrix": cm.tolist(),
    "feature_importances": {k: float(v) for k, v in importances.items()},
}, "/home/claude/titanic_app/titanic_model.pkl")

print("Saved model to titanic_model.pkl")
