# Incident_Triage

## Overview

This project provides a **root cause prediction and incident classification system** for integration error messages using a lightweight **hybrid retrieval approach**.

### How the Model Works (High-Level)

The system is not a traditional machine learning classifier. Instead, it combines three techniques:

#### 1. TF-IDF Retrieval (Primary Engine)

* Converts error messages into numerical vectors using:

  * word-level TF-IDF
  * character-level TF-IDF
* Compares new errors with historical errors from the CSV
* Uses **cosine similarity** to find the closest match
* Returns the root cause explanation from the most similar past error

This is the **core mechanism** for root cause prediction

---

#### 2. Similarity-Based Decision Logic

* If similarity is **high** → use retrieved explanation
* If similarity is **low** → fallback logic is triggered:

  * convert parser errors into business-friendly messages
  * OR use last meaningful sentence from the error

This prevents incorrect outputs when no good match exists

---

#### 3. Anomaly Detection (Isolation Forest)

* Trained on TF-IDF vectors of historical errors
* Detects whether a new error is **structurally different** from known patterns
* Outputs:

  * `is_anomalous` → True/False
  * `anomaly_score`

Helps identify **unusual or new types of errors**

---

#### 4. Incident Classification (Rule-Based)

* Uses keyword matching to assign categories:

  * authentication
  * schema
  * validation
  * timeout
  * mapping
  * duplicate-processing
* Based on both error message and predicted root cause

---

### Hybrid System Summary

```text
Input Error Message
        ↓
TF-IDF Vectorization
        ↓
Similarity Matching (cosine similarity)
        ↓
Root Cause Retrieval (from CSV)
        ↓
Fallback Logic (if similarity low)
        ↓
Incident Classification (rules)
        ↓
Anomaly Detection (Isolation Forest)
```

---

## Features

### 1. Root Cause Prediction

* Uses historical error messages
* Finds most similar past errors
* Returns corresponding explanation

### 2. Smart Fallback Logic

When similarity is low:

* Converts parser errors into business-friendly messages
* Falls back to last meaningful sentence

### 3. Incident Classification

Categories:

* authentication
* schema
* validation
* timeout
* mapping
* duplicate-processing
* other

### 4. Anomaly Detection

* Uses Isolation Forest
* Flags unusual error patterns

### 5. Human Attention Flagging

Triggers when:

* anomaly detected
* similarity is low

---

## Project Structure

```
.
├── root_cause_api_app.py     # FastAPI application
├── train_root_cause_model.py # Training script
├── root_cause_model.pkl      # Trained model (generated)
├── IntegrationErrors.csv     # Training dataset
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install fastapi uvicorn scikit-learn pandas joblib
```

---

### 2. Train the Model

```bash
python train_root_cause_model.py
```

This will generate:

* `root_cause_model.pkl`
* `training_predictions.csv`
* `model_report.md`

---

### 3. Run the API

```bash
python root_cause_api_app.py
```

API runs at:

```
http://localhost:8000
```

---

## API Endpoints

### Health Check

```
GET /health
```

### Predict Root Cause

```
POST /predict
```

#### Request Body

```json
{
  "error_message": "Validation error occurred. Supervisory Organization is not active."
}
```

#### Response

```json
{
  "predicted_root_cause": "Supervisory Organization is not active",
  "incident_category": "validation",
  "similarity_score": 0.82,
  "is_anomalous": false,
  "needs_human_attention": false
}
```

---

## Prediction Logic

### Step 1: Normalize Input

Clean and standardize text

### Step 2: Vectorization

* Word-level TF-IDF
* Character-level TF-IDF

### Step 3: Similarity Matching

Find closest historical error

### Step 4: Root Cause Decision

| Condition       | Action                                    |
| --------------- | ----------------------------------------- |
| High similarity | Use retrieved explanation                 |
| Low similarity  | Convert parser error OR use last sentence |

### Step 5: Classification

Assign incident category using keyword rules

### Step 6: Anomaly Detection

Flag unusual patterns

---

## Parser Error Handling

Example:

Input:

```
Unexpected character '---' (code 32) (missing name?)
```

Output:

```
Invalid value '---' in the Company Ref ID.
```

---

## Configuration

### Similarity Threshold

```python
similarity_alert_threshold = 0.35
```

* Below threshold → fallback logic
* Above threshold → retrieval result

---

## Limitations

* Small dataset reduces anomaly accuracy
* Rule-based classification depends on keywords
* Retrieval may fail for completely new domains

---

## Future Improvements

* Use embeddings instead of TF-IDF
* Train classification model
* Add LLM-based explanation refinement
* Expand training dataset

---

## Summary

This system is a hybrid approach combining:

* Retrieval (TF-IDF similarity)
* Rule-based classification
* Anomaly detection
* Smart fallback logic

Designed for practical, explainable root cause analysis in integration systems.
