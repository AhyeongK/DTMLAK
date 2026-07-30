from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_manufacturing_data.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Load synthetic manufacturing data
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset was not found:\n{DATA_PATH}\n\n"
        "Run generate_data.py first."
    )


df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DTMLAK MODEL TRAINING")
print("=" * 70)

print(f"Dataset path: {DATA_PATH}")
print(f"Number of records: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

print("\nDataset columns:")
print(df.columns.tolist())

print("\nSupport priority distribution:")
print(df["Support_Priority"].value_counts())

# Validate required columns
required_columns = [
    "Operator_ID",
    "Process",
    "Shift",
    "Operator_Level",
    "Training_Score",
    "Experience_Months",
    "Total_Units",
    "Defect_Count",
    "Rework_Count",
    "Support_Priority"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The dataset is missing required columns: "
        f"{missing_columns}"
    )


# Clean and validate numeric values
numeric_columns = [
    "Training_Score",
    "Experience_Months",
    "Total_Units",
    "Defect_Count",
    "Rework_Count"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


if df[numeric_columns].isnull().any().any():
    raise ValueError(
        "One or more numeric columns contain invalid values."
    )


if (df["Total_Units"] <= 0).any():
    raise ValueError(
        "Total_Units must be greater than zero."
    )


# Create model-ready rate features

df["Defect_Rate"] = (
    df["Defect_Count"]
    / df["Total_Units"]
)

df["Rework_Rate"] = (
    df["Rework_Count"]
    / df["Total_Units"]
)


# Select model features
feature_columns = [
    "Process",
    "Shift",
    "Operator_Level",
    "Training_Score",
    "Experience_Months",
    "Defect_Rate",
    "Rework_Rate"
]

target_column = "Support_Priority"

X = df[feature_columns].copy()
y_text = df[target_column].copy()


# Operator_ID is intentionally excluded.
# It is an identifier, not an operational risk factor.


# Encode target labels
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)

print("\nTarget label mapping:")

for label, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(
        label_encoder.classes_
    )
):
    print(f"{label}: {encoded_value}")


# Save the target encoder for the FastAPI backend.
label_encoder_path = (
    MODELS_DIR
    / "support_priority_label_encoder.joblib"
)

joblib.dump(
    label_encoder,
    label_encoder_path
)

print(
    "\nLabel encoder saved to:"
    f"\n{label_encoder_path}"
)

# Define categorical and numeric features
categorical_features = [
    "Process",
    "Shift",
    "Operator_Level"
]

numeric_features = [
    "Training_Score",
    "Experience_Months",
    "Defect_Rate",
    "Rework_Rate"
]

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ],
    remainder="drop"
)

# Create train and test datasets
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nTraining and test split completed.")
print(f"Training records: {len(X_train):,}")
print(f"Test records: {len(X_test):,}")

print("\nTraining target distribution:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nPart 1 completed successfully.")

# Preprocess the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nData preprocessing completed.")

print("Training feature shape:", X_train_processed.shape)
print("Test feature shape:", X_test_processed.shape)

# Train Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(
    X_train_processed,
    y_train
)

rf_predictions = rf_model.predict(
    X_test_processed
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy:.4f}")

# Train XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(
    X_train_processed,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_test_processed
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print("\nXGBoost Accuracy:")
print(f"{xgb_accuracy:.4f}")

# Select Best Model
if rf_accuracy >= xgb_accuracy:

    best_model = rf_model
    best_model_name = "Random Forest"

else:

    best_model = xgb_model
    best_model_name = "XGBoost"

print("\nBest Model Selected:")
print(best_model_name)


# Save trained models and preprocessor
random_forest_model_path = (
    MODELS_DIR
    / "random_forest_model.joblib"
)

xgboost_model_path = (
    MODELS_DIR
    / "xgboost_model.joblib"
)

preprocessor_path = (
    MODELS_DIR
    / "preprocessor.joblib"
)

best_model_path = (
    MODELS_DIR
    / "best_model.joblib"
)


# Save Random Forest
joblib.dump(
    rf_model,
    random_forest_model_path
)

# Save XGBoost
joblib.dump(
    xgb_model,
    xgboost_model_path
)

# Save fitted preprocessing pipeline
joblib.dump(
    preprocessor,
    preprocessor_path
)

# Save the selected best model
joblib.dump(
    best_model,
    best_model_path
)


print("\nModels saved successfully.")

print("\nRandom Forest model:")
print(random_forest_model_path)

print("\nXGBoost model:")
print(xgboost_model_path)

print("\nPreprocessor:")
print(preprocessor_path)

print("\nSelected best model:")
print(best_model_path)