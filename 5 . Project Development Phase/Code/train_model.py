# ==========================================================
# SMART LENDER AI
# Loan Eligibility Prediction System
# ==========================================================

import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# CREATE MODELS FOLDER
# ==========================================================

if not os.path.exists("models"):
    os.makedirs("models")

# ==========================================================
# LOAD DATASET
# ==========================================================

print("="*60)
print("SMART LENDER AI")
print("="*60)

print("\nLoading Dataset...\n")

df = pd.read_csv("loan_data.csv")

# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================

df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully.\n")

print("Columns :\n")

print(df.columns.tolist())

# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("\nDataset Shape :", df.shape)

print("\nFirst Five Rows\n")

print(df.head())

print("\nMissing Values\n")

print(df.isnull().sum())

# ==========================================================
# REMOVE LOAN ID AND PREPROCESS MISSING VALUES
# ==========================================================

if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)
    print("\nLoan_ID removed successfully.")

print("\nHandling missing values...")
# Categorical and discrete columns imputation with mode
for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Continuous columns imputation with median
for col in ["LoanAmount", "Loan_Amount_Term"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

print("Missing values handled successfully.")

# ==========================================================
# LABEL ENCODING
# ==========================================================

print("\nEncoding Categorical Columns...")

encoders = {}

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column].astype(str).str.strip())

    encoders[column] = encoder

print("Encoding Completed Successfully.")

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]

print("\nNumber of Features :", X.shape[1])

print("Number of Samples :", X.shape[0])

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed.")

# ==========================================================
# CREATE MODELS
# ==========================================================

models = {

    "Decision Tree":

        DecisionTreeClassifier(

            random_state=42

        ),

    "Random Forest":

        RandomForestClassifier(

            n_estimators=200,

            random_state=42

        ),

    "KNN":

        KNeighborsClassifier(

            n_neighbors=5

        ),

    "XGBoost":

        XGBClassifier(

            random_state=42,

            eval_metric="logloss"

        )

}

accuracy_scores = {}

best_model = None

best_accuracy = 0

best_model_name = ""

print("\n")

print("="*60)

print("TRAINING MODELS")

print("="*60)

for name, model in models.items():

    print("\n")

    print("="*60)

    print(name)

    print("="*60)

    # KNN requires scaled data

    if name == "KNN":

        model.fit(

            X_train_scaled,

            y_train

        )

        predictions = model.predict(

            X_test_scaled

        )

    else:

        model.fit(

            X_train,

            y_train

        )

        predictions = model.predict(

            X_test

        )

    accuracy = accuracy_score(

        y_test,

        predictions

    )

    accuracy_scores[name] = accuracy

    print(

        "\nAccuracy : {:.2f}%".format(

            accuracy*100

        )

    )

    print("\nClassification Report\n")

    print(

        classification_report(

            y_test,

            predictions

        )

    )

    cm = confusion_matrix(

        y_test,

        predictions

    )

    plt.figure(figsize=(5,4))

    sns.heatmap(

        cm,

        annot=True,

        cmap="Blues",

        fmt="d"

    )

    plt.title(

        f"{name} Confusion Matrix"

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name
    # ==========================================================
# ACCURACY COMPARISON
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

accuracy_df = pd.DataFrame({
    "Model": list(accuracy_scores.keys()),
    "Accuracy": [round(score * 100, 2) for score in accuracy_scores.values()]
})

print("\nModel Accuracy Table\n")
print(accuracy_df)

# ==========================================================
# ACCURACY BAR CHART
# ==========================================================

plt.figure(figsize=(10,6))

bars = plt.bar(
    accuracy_df["Model"],
    accuracy_df["Accuracy"],
    color=["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
)

plt.title(
    "Model Accuracy Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy (%)")

plt.ylim(0,100)

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        height + 1,

        f"{height:.2f}%",

        ha="center",

        fontsize=11

    )

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.show()

# ==========================================================
# BEST MODEL
# ==========================================================

print("\n")
print("=" * 60)

print("BEST MODEL")

print("=" * 60)

print(f"\nModel Name : {best_model_name}")

print(f"Accuracy   : {best_accuracy*100:.2f}%")

# ==========================================================
# SAVE MODEL
# ==========================================================

print("\nSaving Best Model...")

joblib.dump(

    best_model,

    "models/model.pkl"

)

print("model.pkl saved successfully.")

# ==========================================================
# SAVE ENCODERS
# ==========================================================

joblib.dump(

    encoders,

    "models/encoder.pkl"

)

print("encoder.pkl saved successfully.")

# ==========================================================
# SAVE SCALER
# ==========================================================

joblib.dump(

    scaler,

    "models/scaler.pkl"

)

print("scaler.pkl saved successfully.")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\nGenerating Feature Importance...")

if best_model_name in ["Decision Tree", "Random Forest", "XGBoost"]:

    importance = best_model.feature_importances_

    features = X.columns

    feature_df = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(

        by="Importance",

        ascending=False

    )

    print("\nTop Important Features\n")

    print(feature_df)

    plt.figure(figsize=(12,6))

    plt.bar(

        feature_df["Feature"],

        feature_df["Importance"],

        color="#16a085"

    )

    plt.xticks(rotation=45)

    plt.title(

        f"{best_model_name} Feature Importance",

        fontsize=15,

        fontweight="bold"

    )

    plt.ylabel("Importance")

    plt.tight_layout()

    plt.show()

# ==========================================================
# SAMPLE PREDICTION
# ==========================================================

print("\nRunning Sample Prediction...\n")

sample = X.iloc[[0]]

if best_model_name == "KNN":

    sample_scaled = scaler.transform(sample)

    prediction = best_model.predict(sample_scaled)

    probability = best_model.predict_proba(sample_scaled)

else:

    prediction = best_model.predict(sample)

    probability = best_model.predict_proba(sample)

prediction_label = encoders["Loan_Status"].inverse_transform(prediction)[0]

confidence = probability.max() * 100

print("Prediction :", prediction_label)

print(f"Confidence : {confidence:.2f}%")

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

print("\n")
print("=" * 60)

print("SMART LENDER AI PROJECT SUMMARY")

print("=" * 60)

print(f"""
Dataset Records      : {len(df)}

Training Samples     : {len(X_train)}

Testing Samples      : {len(X_test)}

Best Model           : {best_model_name}

Best Accuracy        : {best_accuracy*100:.2f}%

Model Saved          : models/model.pkl

Encoder Saved        : models/encoder.pkl

Scaler Saved         : models/scaler.pkl
""")

print("=" * 60)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 60)