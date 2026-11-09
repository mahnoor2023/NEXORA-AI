from __future__ import annotations

import json
import math
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
TEST_FRACTION = 0.2
MAX_ML_ROWS = 20_000
MAX_ML_COLUMNS = 50
MAX_CLASSES = 50
MAX_PREDICTIONS = 50
MAX_IMPORTANCE_FEATURES = 12
MAX_TRAIN_SECONDS = 20


class MLValidationError(ValueError):
    """An uploaded dataset or requested task cannot be trained safely."""


def _detect_task(target: pd.Series) -> str:
    unique = target.nunique(dropna=True)
    if is_numeric_dtype(target) and not is_bool_dtype(target):
        # Do not auto-classify integer identifiers or continuous targets whose
        # class count exceeds the supported classifier limit.
        return "classification" if unique <= 15 or (unique <= MAX_CLASSES and unique / max(len(target), 1) <= 0.05) else "regression"
    return "classification"


def _json_scalar(value: Any) -> str | float | int | bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    max_categories=50,
                    sparse_output=True,
                ),
            ),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def _feature_importance(pipeline: Pipeline, model: Any, source_columns: list[str]) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importances = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    else:
        return []

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    grouped: dict[str, float] = {}
    source_columns_by_length = sorted(source_columns, key=len, reverse=True)
    for name, importance in zip(names, importances, strict=False):
        transformed_name = str(name)
        column = next(
            (candidate for candidate in source_columns_by_length if transformed_name.startswith(f"numeric__{candidate}") or transformed_name.startswith(f"categorical__{candidate}_") or transformed_name == f"categorical__{candidate}"),
            transformed_name.split("__", 1)[-1],
        )
        grouped[column] = grouped.get(column, 0.0) + float(importance)
    return [
        {"feature": name, "importance": round(value, 6)}
        for name, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:MAX_IMPORTANCE_FEATURES]
    ]


def _rule_based_insights(
    *, task: str, target: str, total_rows: int, training_rows: int, test_rows: int,
    missing_target_rows: int, best_model: dict[str, Any],
    importance: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = best_model["metrics"]
    if task == "classification":
        score_name = "f1_weighted"
        score = metrics[score_name]
        score_text = f"weighted F1 {score:.3f} and accuracy {metrics['accuracy']:.3f}"
    else:
        score_name = "r2"
        score = metrics[score_name]
        score_text = f"R² {score:.3f}, MAE {metrics['mae']:.3f}, and RMSE {metrics['rmse']:.3f}"
    bullets = [
        f"The target is {target!r}; the split used {training_rows:,} training rows and {test_rows:,} held-out rows from {total_rows:,} usable records.",
        f"{best_model['name']} scored {score_text} on the held-out set.",
    ]
    if missing_target_rows:
        bullets.append(f"{missing_target_rows:,} rows with a missing target were excluded before splitting.")
    if importance:
        bullets.append(f"The strongest model-supported feature signal was {importance[0]['feature']!r}; this is predictive importance, not evidence of causation.")
    bullets.append("Scores come from one fixed-seed holdout split; they do not establish future or causal performance.")
    return {
        "provider": "rule_based",
        "label": "Rule-based insights · not LLM-generated",
        "title": "Evidence from this training run",
        "summary": f"For this {task} run, {best_model['name']} ranked highest by {score_name} on the held-out rows. Treat the result as a baseline, not a guarantee of performance on new data.",
        "bullets": bullets,
        "limitations": [
            "A single random holdout can vary; cross-validation and external validation were not run.",
            "Feature importance describes model use, not causation or a verified explanation.",
            "Potential time, group, or domain leakage beyond exact target-copy columns cannot be detected automatically.",
        ],
    }


def _llm_insights(evidence: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    api_url = os.getenv("LLM_API_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not any((api_key, api_url, model)):
        return fallback
    if not (api_key and api_url and model):
        return {**fallback, "fallback_reason": "LLM configuration is incomplete; set LLM_API_KEY, LLM_API_URL, and LLM_MODEL on the server."}
    if not api_url.startswith(("https://", "http://")):
        return {**fallback, "fallback_reason": "LLM_API_URL must use HTTP or HTTPS; showing rule-based insights."}
    request_data = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Summarize only the evidence in the supplied JSON. Do not invent facts, causal claims, or recommendations. State when evidence is limited. Return concise plain text."},
            {"role": "user", "content": json.dumps(evidence, allow_nan=False)},
        ],
    }
    request = Request(
        api_url,
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read(64_000))
        text = payload["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            return fallback
        return {**fallback, "provider": "llm", "label": f"LLM-assisted · {model}", "summary": text.strip(), "bullets": []}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
        return {**fallback, "fallback_reason": "Configured LLM was unavailable; showing rule-based insights."}


def train_and_evaluate(frame: pd.DataFrame, target: str, requested_task: str = "auto") -> dict[str, Any]:
    started = time.monotonic()
    if requested_task not in {"auto", "classification", "regression"}:
        raise MLValidationError("Task must be auto, classification, or regression.")
    if len(frame) > MAX_ML_ROWS:
        raise MLValidationError(f"Training is limited to {MAX_ML_ROWS:,} rows.")
    if len(frame.columns) > MAX_ML_COLUMNS:
        raise MLValidationError(f"Training is limited to {MAX_ML_COLUMNS} columns.")
    if target not in frame.columns:
        raise MLValidationError("Choose a target column that exists in the uploaded CSV.")
    if frame.shape[1] < 2:
        raise MLValidationError("Training requires at least one feature column in addition to the target.")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    original_target = frame[target]
    usable = frame.loc[original_target.notna()].copy()
    y = usable[target]
    if len(usable) < 20:
        raise MLValidationError(f"Only {len(usable)} rows have a non-missing target; at least 20 are required for a train/test evaluation. Add more complete rows or choose another target.")
    if y.nunique(dropna=True) < 2:
        raise MLValidationError("The target must contain at least two distinct values.")

    task = _detect_task(y) if requested_task == "auto" else requested_task
    if task == "regression" and (not is_numeric_dtype(y) or is_bool_dtype(y)):
        raise MLValidationError("Regression requires a numeric target. Choose classification or a numeric target column.")
    if task == "classification":
        class_count = int(y.nunique())
        class_sizes = y.value_counts(dropna=True)
        test_rows = math.ceil(len(y) * TEST_FRACTION)
        if class_count > MAX_CLASSES:
            raise MLValidationError(f"Classification target has {class_count} distinct classes; the limit is {MAX_CLASSES}. This may be a high-cardinality label or identifier. Choose a lower-cardinality target, group rare labels, or select regression for a continuous numeric target.")
        smallest_class = int(class_sizes.min())
        if smallest_class < 2:
            raise MLValidationError(f"The rarest class has only {smallest_class} row. Stratified evaluation requires at least two rows per class; collect more examples, combine justified rare labels, or choose another target.")
        if test_rows < class_count or len(y) - test_rows < class_count:
            raise MLValidationError(f"The {len(y)} usable rows produce only {test_rows} test rows, which is too few to represent all {class_count} classes in both split partitions. Use fewer classes or provide more examples per class.")

    features = [column for column in usable.columns if column != target]
    exact_copies = [column for column in features if usable[column].notna().all() and usable[column].eq(y).all()]
    if exact_copies:
        raise MLValidationError(
            "Potential target leakage detected: feature columns exactly copy the selected target ("
            + ", ".join(map(str, exact_copies[:5]))
            + "). Remove these columns before training."
        )
    if not features:
        raise MLValidationError("Training requires at least one feature column besides the target.")

    X = usable[features]
    numeric = [column for column in features if is_numeric_dtype(X[column]) and not is_bool_dtype(X[column])]
    categorical = [column for column in features if column not in numeric]
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=y
        )
        models: list[tuple[str, Any]] = [
            ("Logistic regression", LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
            ("Random forest", RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=RANDOM_SEED)),
        ]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED
        )
        models = [
            ("Ridge regression", Ridge()),
            ("Random forest", RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=RANDOM_SEED)),
        ]

    results: list[dict[str, Any]] = []
    labels = sorted(pd.unique(y).tolist(), key=lambda value: str(value)) if task == "classification" else []
    for model_name, estimator in models:
        pipeline = Pipeline(steps=[("preprocessor", _build_preprocessor(numeric, categorical)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        if task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            }
            matrix = confusion_matrix(y_test, predictions, labels=pipeline.named_steps["model"].classes_).tolist()
        else:
            metrics = {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
            }
            matrix = None
        importance = _feature_importance(pipeline, pipeline.named_steps["model"], features)
        result = {
            "name": model_name,
            "metrics": metrics,
            "confusion_matrix": matrix,
            "class_labels": [_json_scalar(value) for value in pipeline.named_steps["model"].classes_] if task == "classification" else [],
            "predictions": [
                {"actual": _json_scalar(actual), "predicted": _json_scalar(predicted)}
                for actual, predicted in list(zip(y_test.tolist(), predictions.tolist(), strict=False))[:MAX_PREDICTIONS]
            ],
            "feature_importance": importance,
        }
        results.append(result)
        if time.monotonic() - started > MAX_TRAIN_SECONDS:
            raise MLValidationError(f"Training exceeded the {MAX_TRAIN_SECONDS}-second execution limit.")

    metric_name = "f1_weighted" if task == "classification" else "r2"
    best_result = max(results, key=lambda result: result["metrics"][metric_name])
    train_rows = len(X_train)
    test_rows = len(X_test)
    missing_target_rows = len(frame) - len(usable)
    fallback = _rule_based_insights(
        task=task, target=target, total_rows=len(usable), training_rows=train_rows,
        test_rows=test_rows, missing_target_rows=missing_target_rows,
        best_model=best_result,
        importance=best_result["feature_importance"],
    )
    evidence = {
        "task": task,
        "target": target,
        "usable_rows": len(usable),
        "training_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": len(features),
        "best_model": best_result["name"],
        "best_model_metrics": best_result["metrics"],
        "top_feature_importance": best_result["feature_importance"][:5],
        "missing_target_rows_excluded": missing_target_rows,
    }
    insights = _llm_insights(evidence, fallback)
    if time.monotonic() - started > MAX_TRAIN_SECONDS + 10:
        raise MLValidationError("Training and insights exceeded the execution time limit.")
    return {
        "task": task,
        "target": target,
        "random_seed": RANDOM_SEED,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "usable_rows": len(usable),
        "missing_target_rows": missing_target_rows,
        "feature_count": len(features),
        "features": [str(column) for column in features],
        "models": results,
        "best_model": best_result["name"],
        "primary_metric": metric_name,
        "class_labels": [_json_scalar(value) for value in labels],
        "insights": insights,
        "limitations": [
            "One fixed-seed train/test split was used; no cross-validation, hyperparameter search, or calibration was performed.",
            "Exact copies of the target are rejected. Time-based, group-based, proxy, and domain-specific leakage require human review.",
            "Metrics are baselines and do not guarantee performance on future or external data.",
        ],
        "training_seconds": round(time.monotonic() - started, 3),
    }
