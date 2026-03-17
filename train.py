import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("loan_approval_data.csv")

df = df.drop("Applicant_ID", axis=1)

# ⭐ IMPORTANT FIX
df = df.dropna(subset=["Loan_Approved"])

X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Pipeline Model Saved ✅")