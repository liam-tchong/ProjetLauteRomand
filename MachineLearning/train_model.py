import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print("Loading data...")
df = pd.read_csv("match_data.csv")
print(f"Columns available: {list(df.columns)}")

# The key columns from football-data.co.uk
# FTHG = Full Time Home Goals, FTAG = Full Time Away Goals, FTR = Full Time Result
df = df[["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HF", "AF"]].dropna()

# Convert result to numbers: H=0 (home win), D=1 (draw), A=2 (away win)
df["result"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})

# Features
X = df[["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF"]]
y = df["result"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc:.2%}")

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")