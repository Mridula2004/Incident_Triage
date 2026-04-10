import re
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_FILE_CANDIDATES = [
    "IntegrationErrors(Sheet1).csv",
    "training_data.csv",
    "data.csv",
]

MODEL_PATH = "root_cause_model.pkl"
REPORT_PATH = "model_report.md"
PREDICTIONS_PATH = "training_predictions.csv"


def find_data_file() -> Path:
    for candidate in DATA_FILE_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find training CSV. Expected one of: "
        + ", ".join(DATA_FILE_CANDIDATES)
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Failed to read CSV: {last_error}")

    required_columns = {"Error Message", "Explanation"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Error Message"] = df["Error Message"].fillna("").astype(str).str.strip()
    df["Explanation"] = df["Explanation"].fillna("").astype(str).str.strip()

    df = df[(df["Error Message"] != "") & (df["Explanation"] != "")].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after cleaning the CSV.")

    return df


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def classify_incident_category(error_message: str, predicted_root_cause: str) -> str:
    text = f"{error_message} {predicted_root_cause}".lower()

    if any(term in text for term in [
        "auth", "authentication", "unauthorized", "forbidden", "invalid credential",
        "access denied", "login", "oauth", "token expired", "bad credentials"
    ]):
        return "authentication"

    if any(term in text for term in [
        "schema", "xsd", "xml parse", "json parse", "invalid xml", "invalid json",
        "field missing", "element", "datatype", "payload structure"
    ]):
        return "schema"

    if any(term in text for term in [
        "validation", "not active", "inactive", "not valid", "cannot be rescinded",
        "effective date", "future-dated", "future dated", "pending transaction",
        "no longer available", "not available", "invalid value", "business process"
    ]):
        return "validation"

    if any(term in text for term in [
        "timeout", "timed out", "time out", "socket timeout", "read timeout",
        "connection timeout", "gateway timeout"
    ]):
        return "timeout"

    if any(term in text for term in [
        "mapping", "mapped", "transformation", "crosswalk", "lookup failed",
        "translation", "source value", "target value"
    ]):
        return "mapping"

    if any(term in text for term in [
        "duplicate", "already exists", "already processed", "reprocessed",
        "duplicate-processing", "duplicate processing", "idempotent", "same event"
    ]):
        return "duplicate-processing"

    return "other"


def main() -> None:
    csv_path = find_data_file()
    df = load_data(csv_path)

    df["normalized_error_message"] = df["Error Message"].apply(normalize_text)

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )

    Xw = word_vectorizer.fit_transform(df["normalized_error_message"])
    Xc = char_vectorizer.fit_transform(df["normalized_error_message"])
    X = hstack([Xw, Xc])

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
    )
    anomaly_model.fit(X.toarray())

    incident_categories = [
        classify_incident_category(err, exp)
        for err, exp in zip(df["Error Message"], df["Explanation"])
    ]
    df["incident_category"] = incident_categories

    artifact = {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "X": X,
        "df": df[
            [
                "Error Message",
                "Explanation",
                "normalized_error_message",
                "incident_category",
            ]
        ].copy(),
        "anomaly_model": anomaly_model,
        "similarity_alert_threshold": 0.35,
        "model_type": "tfidf_retrieval_plus_anomaly_detection",
        "version": "2.0",
    }

    joblib.dump(artifact, MODEL_PATH)

    training_predictions = pd.DataFrame({
        "Error Message": df["Error Message"],
        "Actual Explanation": df["Explanation"],
        "Predicted Explanation": df["Explanation"],
        "Incident Category": df["incident_category"],
    })
    training_predictions.to_csv(PREDICTIONS_PATH, index=False)

    report = f"""# Root Cause Model Report

## Summary
- Training file: {csv_path.name}
- Rows used for training: {len(df)}
- Model type: TF-IDF retrieval + Isolation Forest anomaly detector
- Word n-grams: (1, 2)
- Character n-grams: (3, 5)
- Similarity alert threshold: 0.35
- Isolation Forest contamination: 0.10

## Output Fields
- predicted_root_cause
- incident_category
- similarity_score
- is_anomalous
- anomaly_score
- needs_human_attention
- alert_reasons
- top_matches

## Incident Category Labels
- authentication
- schema
- validation
- timeout
- mapping
- duplicate-processing
- other

## Notes
- Root cause prediction is retrieval-based.
- Incident category is rule-based using error text plus matched explanation.
- Anomaly detection flags incidents materially different from known historical patterns.
- With small datasets, anomaly detection should be treated as a review signal, not absolute truth.
"""
    Path(REPORT_PATH).write_text(report, encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "model_path": MODEL_PATH,
        "rows_trained": len(df),
        "report_path": REPORT_PATH,
        "predictions_path": PREDICTIONS_PATH,
    }, indent=2))


if __name__ == "__main__":
    main()
