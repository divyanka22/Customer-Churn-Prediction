import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

# Load the customer churn dataset
data = pd.read_csv("data/telco_churn.csv")


# Clean TotalCharges column

# Replace blank strings with missing values
data["TotalCharges"] = data["TotalCharges"].replace(
    r"^\s*$", np.nan, regex=True
)

# Convert TotalCharges into numeric format
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

# Check missing values again
print("\nMISSING VALUES AFTER CONVERSION:")
print(data.isnull().sum())

# Check customers with missing TotalCharges
print("\nROWS WITH MISSING TOTAL CHARGES:")
print(data[data["TotalCharges"].isnull()][["tenure", "TotalCharges"]])

# Fill missing TotalCharges with 0
data["TotalCharges"] = data["TotalCharges"].fillna(0)


# FEATURE ENGINEERING

# Feature 1: Categorize customers by tenure
data["TenureGroup"] = pd.cut(
    data["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Short-term", "Medium-term", "Long-term"]
)

# Feature 2: Count total services used
service_columns = [
    "PhoneService",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

data["TotalServices"] = (
    (data["PhoneService"] == "Yes").astype(int) +
    (data["InternetService"] != "No").astype(int) +
    (data[service_columns[2:]] == "Yes").sum(axis=1)
)

# Feature 3: Identify long-term customers
data["IsLongTermCustomer"] = (
    data["tenure"] >= 24
).astype(int)

# Feature 4: Calculate average monthly spend
data["AverageMonthlySpend"] = np.where(
    data["tenure"] > 0,
    data["TotalCharges"] / data["tenure"],
    data["MonthlyCharges"]
)


# SEPARATE FEATURES AND TARGET

# Remove customerID because it is only an identifier
X = data.drop(columns=["customerID", "Churn"])

# Convert churn labels into binary values
y = data["Churn"].map({"No": 0, "Yes": 1})

# Identify numerical columns
numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServices",
    "IsLongTermCustomer",
    "AverageMonthlySpend"
]

# Identify categorical columns
categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# MODEL TRAINING AND HYPERPARAMETER TUNING

# Create three machine learning pipelines
models = {
    "Logistic Regression": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ]),
        {
            "model__C": [0.1, 1, 10]
        }
    ),

    "Random Forest": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20, None]
        }
    ),

    "Decision Tree": (
        Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ))
        ]),
        {
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 10]
        }
    )
}

# Store model results
results = {}

# Train and tune each model
for name, (pipeline, parameters) in models.items():

    print(f"\nTraining {name}...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Predict churn probabilities
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    results[name] = {
        "model": best_model,
        "best_params": grid_search.best_params_,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
        "pr_auc": pr_auc
    }

    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("ROC-AUC:", round(auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Compare all models
print("\n========== MODEL COMPARISON ==========")

for name, result in results.items():
    print(
        name,
        "| Accuracy:", round(result["accuracy"], 4),
        "| Precision:", round(result["precision"], 4),
        "| Recall:", round(result["recall"], 4),
        "| ROC-AUC:", round(result["roc_auc"], 4)
    )

# Select the best model using ROC-AUC
best_model_name = max(
    results,
    key=lambda name: results[name]["roc_auc"]
)

best_model = results[best_model_name]["model"]

print("\nBEST MODEL:", best_model_name)
print("BEST ROC-AUC:", results[best_model_name]["roc_auc"])

# Save the best trained model

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_churn_model.joblib"
)

print("\nBest model saved successfully!")
print("Model name:", best_model_name)
print("Saved location: models/best_churn_model.joblib")

# Fit on training data and transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Display results
print("\nPREPROCESSING RESULTS:")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

print("\nTARGET DISTRIBUTION:")
print(y_train.value_counts())


# Display newly created features
print("\nENGINEERED FEATURES:")
print(data[[
    "TenureGroup",
    "TotalServices",
    "IsLongTermCustomer",
    "AverageMonthlySpend"
]].head())

print("\nUPDATED DATASET SHAPE:")
print(data.shape)


# Verify cleaning
print("\nDATA TYPES AFTER CLEANING:")
print(data.dtypes)

print("\nMISSING VALUES AFTER FILLING:")
print(data.isnull().sum())


# Display the first five records
print("FIRST FIVE RECORDS:")
print(data.head())

# Display dataset dimensions
print("\nDATASET SHAPE:")
print(data.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(data.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION:")
data.info()


# Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(data.duplicated().sum())

# Check data types
print("\nDATA TYPES:")
print(data.dtypes)

# Check statistical summary
print("\nSTATISTICAL SUMMARY:")
print(data.describe())

# Check churn distribution
print("\nCHURN DISTRIBUTION:")
print(data["Churn"].value_counts())
