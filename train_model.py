
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATA_PATH = "data/ecommerce_returns.csv"


# STEP 1: Load and preprocess dataset

def prepare_data(df):

    df = df.copy()

    numeric_columns = [
        "Product_Price_INR",
        "Discount_Percent",
        "Expected_Delivery_Days",
        "Actual_Delivery_Days",
        "Customer_Rating_PreOrder",
        "Previous_Returns",
        "Previous_Orders"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Handle invalid values

    df.loc[
        df["Product_Price_INR"] <= 0,
        "Product_Price_INR"
    ] = np.nan

    df.loc[
        ~df["Discount_Percent"].between(0, 100),
        "Discount_Percent"
    ] = np.nan

    df.loc[
        df["Expected_Delivery_Days"] <= 0,
        "Expected_Delivery_Days"
    ] = np.nan

    df.loc[
        df["Actual_Delivery_Days"] <= 0,
        "Actual_Delivery_Days"
    ] = np.nan

    df.loc[
        ~df["Customer_Rating_PreOrder"].between(1, 5),
        "Customer_Rating_PreOrder"
    ] = np.nan

    df.loc[
        df["Previous_Returns"] < 0,
        "Previous_Returns"
    ] = np.nan

    df.loc[
        df["Previous_Orders"] <= 0,
        "Previous_Orders"
    ] = np.nan

    df.loc[
        df["Previous_Returns"] > df["Previous_Orders"],
        "Previous_Returns"
    ] = np.nan

    # STEP 2: Feature engineering

    df["Delivery_Delay"] = (
        df["Actual_Delivery_Days"] -
        df["Expected_Delivery_Days"]
    )

    df["Customer_Return_Frequency"] = (
        df["Previous_Returns"] /
        df["Previous_Orders"]
    )

    df["Discount_Amount"] = (
        df["Product_Price_INR"] *
        df["Discount_Percent"] / 100
    )

    # Remove unnecessary order ID

    df.drop(
        columns=["Order_ID"],
        inplace=True,
        errors="ignore"
    )

    # STEP 3: Encode target variable

    df["Return_Status"] = df["Return_Status"].map({
        "Not Returned": 0,
        "Returned": 1
    })

    df.dropna(
        subset=["Return_Status"],
        inplace=True
    )

    df = df.replace(
        [np.inf, -np.inf], np.nan
    )

    return df


# STEP 4: Train models

def train():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    df = prepare_data(df)

    print("After preprocessing:", df.shape)

    X = df.drop(columns=["Return_Status"])
    y = df["Return_Status"].astype(int)

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    }

    results = {}

    best_score = -1
    best_model = None
    best_name = None

    for name, classifier in models.items():

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        print("\nTraining:", name)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test, predictions
        )

        precision = precision_score(
            y_test, predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test, predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test, predictions,
            zero_division=0
        )

        results[name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        if f1 > best_score:
            best_score = f1
            best_model = pipeline
            best_name = name

    # STEP 5: Save best model

    os.makedirs("model", exist_ok=True)

    joblib.dump(
        best_model,
        "model/return_model.pkl"
    )

    with open("model/metrics.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "results": results
        }, f, indent=4)

    print("\nModel Comparison:", results)

    print("\nBest Model:", best_name)

    print("\nModel saved successfully!")


if __name__ == "__main__":
    train()
