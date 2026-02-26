# Routing Decision Mechanism (ML + LLM Fallback)

## Why this matters in the demo

Assort Design uses a **hybrid routing strategy** to determine the target audience for each document.

There are two routing paths:

- **ML-first routing** using a TF-IDF + Logistic Regression classifier
- **LLM fallback routing** when the ML router is uncertain (low confidence or ambiguity)

This is a practical agentic design choice:

- use **fast, low-cost ML routing** when confidence is strong
- escalate to **LLM reasoning** when the ML result is uncertain

---

## Routing behavior (high level)

### Primary path: ML router
The ML router attempts to identify the audience first (e.g., `commercial`, `medical_affairs`, `r_and_d`).

### Fallback path: LLM router
If the ML router is not confident enough, the pipeline falls back to the LLM for the final routing decision.

### Current confidence threshold
- If ML top confidence is below the configured threshold (default: **0.58**, set in `agent_profiles.yaml`), the request is sent to the **LLM router**.

### Ambiguity guardrail (margin rule)
- If the top two ML class probabilities are too close, the result is treated as ambiguous and can also fall back to the LLM.

---

## Where the routing logic lives

### 1) Training pipeline — `app/train_router.py`
This file trains the ML router and saves the artifacts used at inference time.

### What it does
- Loads labeled examples from the database:
  - `document.content` (input text)
  - `job.audience` (label)
- Builds a scikit-learn pipeline:
  - `TfidfVectorizer` (unigrams + bigrams, sublinear TF, up to ~30k features)
  - `LogisticRegression` (lbfgs solver, balanced class weights)
- Splits data into train/test (80/20 stratified)
- Evaluates model performance (accuracy + per-class precision/recall/F1)
- Saves artifacts to `artifacts/`:
  - `vectorizer.pkl`
  - `classifier.pkl`
  - `metadata.json`

---

### 2) Inference logic — `app/ml_router.py`
This file loads the trained artifacts and makes the runtime routing decision.

### What it does
- `load()`
  - lazily loads `.pkl` artifacts into memory
- `predict()`
  - vectorizes incoming text
  - computes class probabilities
  - applies threshold + margin decision rules

### Core decision rules
- If top probability (**p1**) is below threshold → treat as **uncertain**
- If top two probabilities are too close (**p1 - p2 < margin**) → treat as **ambiguous**
- Otherwise → use the top predicted class directly

When uncertain/ambiguous, the router does **not** finalize a specialist audience and instead signals fallback behavior.

---

### 3) Pipeline integration — `app/graph.py`
This file integrates ML routing with LLM fallback inside the routing node.

### Inside `route_audience_node()`
The routing flow is:

1. Try loading ML artifacts
2. If artifacts exist → run ML prediction
3. If ML is confident → use ML result directly (`routing_source = "ml"`)
4. If ML is uncertain/ambiguous → fall back to LLM (`routing_source = "ml+llm_fallback"`)
5. If no ML artifacts exist → use LLM routing only (`routing_source = "llm"`)

This creates a **hybrid routing orchestration** pattern:
- deterministic first-pass routing with ML
- reasoning-based fallback with LLM

---

## How ML confidence is computed (p1 and p2)

The ML router uses class probabilities from Logistic Regression to decide whether to trust its own prediction.

### Step 1: Convert text to TF-IDF features
The incoming document text is transformed into a sparse feature vector using the trained `TfidfVectorizer`.

- uses the vocabulary learned during training
- includes unigrams and bigrams
- produces a numeric feature vector (not raw text matching)

---

### Step 2: Logistic Regression outputs class probabilities
The classifier computes a score for each audience class and converts those scores to probabilities.

Conceptually (softmax over class scores):

\[
P(class_i) = \frac{e^{W_i \cdot X + b_i}}{\sum_j e^{W_j \cdot X + b_j}}
\]

Example output:
- `commercial: 0.12`
- `medical_affairs: 0.75`
- `r_and_d: 0.13`

---

### Step 3: Extract top two probabilities
The router sorts probabilities in descending order and selects:

- **p1** = highest probability (top predicted class)
- **p2** = second-highest probability (runner-up)

Example:
- `p1 = 0.75` (`medical_affairs`)
- `p2 = 0.13` (`r_and_d`)

---

### Step 4: Apply routing guardrails
The router checks:

- **Confidence rule:** `p1 < threshold`
- **Ambiguity rule:** `(p1 - p2) < margin`

If either rule is triggered, the ML result is treated as **not reliable enough** for final routing, and the pipeline falls back to the LLM.

---

## What is actually stored in the ML artifacts (`.pkl`)

This is an important clarification.

### `vectorizer.pkl` stores
- learned vocabulary
- tokenization / TF-IDF transformation logic
- feature mapping rules

✅ It **does not** store original training documents for direct text matching.

---

### `classifier.pkl` stores
- learned logistic regression coefficients (weights)
- intercepts
- class mapping / model state

✅ It **does not** store text samples.

---

## What happens at runtime (inference summary)

When a new document arrives:

1. `vectorizer.pkl` transforms the text into a TF-IDF feature vector
2. `classifier.pkl` computes class scores using learned weights
3. scores are converted into probabilities (e.g., `p1`, `p2`, `p3`)
4. threshold/margin guardrails decide whether to:
   - accept the ML route, or
   - fall back to the LLM router

### Key idea
This is **mathematical pattern recognition**, not document-to-document text matching.

---

## Auto-retraining behavior (production / VPS)

The ML router can retrain automatically in production after enough new completed jobs.

### Trigger
After each **completed job**, `graph.py` calls `_maybe_retrain(session)`.

### What `_maybe_retrain()` checks
It compares:
- total completed jobs in the database
- `n_docs` stored in `artifacts/metadata.json` (documents used in the last training run)

If the difference is greater than or equal to the retrain interval (default: **20**), retraining is triggered.

### After retraining
The code calls `_ml_router.reload()` so the updated model is loaded into memory immediately (no app restart required).

---

## Practical routing modes over time

### Before ML artifacts exist
- **LLM-only routing**

### After initial ML training artifacts exist
- **ML-first routing with LLM fallback**

### As more labeled jobs accumulate
- **Automatic retraining** every N completed jobs (default `20`)
- **Hot reload** of updated ML artifacts

---

## Configuration notes

- `RETRAIN_EVERY_N_JOBS=20` → retrain every 20 completed jobs (default)
- `RETRAIN_EVERY_N_JOBS=0` → disable auto-retraining

The routing threshold and margin values are configured through app config and used by the ML router during prediction.

---

## Concurrency / safety note

A lock in `graph.py` prevents duplicate retraining from concurrent requests, which helps avoid overlapping retrain runs in a VPS / multi-user environment.

---

## Summary (routing-only)

Assort Design uses an **ML-first, LLM-fallback routing mechanism**:

- ML handles fast audience classification when confidence is high
- LLM handles uncertain or ambiguous cases
- Routing behavior is controlled by confidence and margin guardrails
- The ML router can retrain automatically as new completed jobs accumulate

This creates a routing layer that is both **efficient** and **adaptive**, while preserving LLM reasoning for edge cases.
