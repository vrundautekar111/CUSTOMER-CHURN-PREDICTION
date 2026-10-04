import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

DATA_PATH = "dataset/customer_churn.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------

df = df.drop_duplicates()

# Convert total charges to numeric
df["total_charges"] = pd.to_numeric(
    df["total_charges"],
    errors="coerce"
)

# Remove rows where target is missing
df = df.dropna(subset=["churn"])


# --------------------------------------------------
# 3. TARGET VARIABLE
# --------------------------------------------------

df["churn"] = df["churn"].map({
    "Yes": 1,
    "No": 0
})

X = df.drop("churn", axis=1)
y = df["churn"]


# --------------------------------------------------
# 4. IDENTIFY COLUMNS
# --------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numeric_columns)


# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# --------------------------------------------------
# 6. MACHINE LEARNING MODEL
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 7. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining model...")


# --------------------------------------------------
# 8. TRAIN
# --------------------------------------------------

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed!")


# --------------------------------------------------
# 9. PREDICTION
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 10. EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "model/churn_model.pkl"
)

print(
    "\nModel saved successfully!"
)

print(
    "Location: model/churn_model.pkl"
)