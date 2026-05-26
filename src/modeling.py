import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(df, target, drop_columns=None):
    """Split data into features and target."""
    drop_columns = drop_columns or []
    default_drop = ["CustomerID", target]
    columns_to_drop = [
        col
        for col in default_drop + drop_columns
        if col in df.columns
    ]
    X = df.drop(columns=columns_to_drop)
    datetime_columns = X.select_dtypes(include=["datetime64[ns]"]).columns
    for column in datetime_columns:
        X[column] = X[column].dt.strftime("%Y-%m-%d")
    y = df[target]
    return X, y


def build_preprocessor(X):
    """Create preprocessing for mixed insurance data."""
    numeric_features = (
        X.select_dtypes(include=["number", "bool"])
        .columns
        .tolist()
    )
    categorical_features = (
        X.select_dtypes(exclude=["number", "bool"])
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def regression_models(random_state=42):
    """Return the required severity-regression models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except ImportError:
        pass

    return models


def classification_models(random_state=42):
    """Return claim-probability models for the pricing framework."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_regression_models(df, target="TotalClaims", random_state=42):
    """Train and score severity models on policies with positive claims."""
    severity_df = df[df[target] > 0].copy()
    leakage = ["ClaimAmount", "Claimed", "ClaimFrequencyFlag", "LossRatio"]
    X, y = prepare_features(severity_df, target=target, drop_columns=leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)

    results = []
    fitted_models = {}
    for name, model in regression_models(random_state).items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        results.append(
            {
                "model": name,
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
            }
        )
        fitted_models[name] = pipe

    return pd.DataFrame(results).sort_values("rmse"), fitted_models


def evaluate_claim_classifier(df, random_state=42):
    """Train and score binary claim-probability models."""
    data = df.copy()
    data["ClaimFlag"] = (data["TotalClaims"] > 0).astype(int)
    leakage = ["TotalClaims", "ClaimAmount", "Claimed", "LossRatio"]
    X, y = prepare_features(data, target="ClaimFlag", drop_columns=leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)

    results = []
    fitted_models = {}
    for name, model in classification_models(random_state).items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        fitted_models[name] = pipe

    return (
        pd.DataFrame(results).sort_values("f1", ascending=False),
        fitted_models,
    )


def risk_based_premium(
    probability,
    severity,
    expense_loading=0.10,
    profit_margin=0.15,
):
    """Combine claim probability and severity into an indicated premium."""
    expected_loss = probability * severity
    return expected_loss * (1 + expense_loading + profit_margin)


def shap_summary_values(fitted_pipeline, X_sample):
    """Return SHAP values for a fitted sklearn pipeline."""
    import shap

    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    return shap_values, transformed, feature_names
