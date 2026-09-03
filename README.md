# E-Commerce Sentiment Analysis & Triage Pipeline

## Problem Statement
Can customer sentiment be accurately classified from unstructured review text at scale, and what latent themes drive negative feedback?

## Dataset
- **Source:** HuggingFace Hub (`mteb/amazon_reviews_multi`)
- **Size:** 10,000 rows (sampled) × 4 features
- **Description:** Amazon product reviews (English subset) with 1-5 star ratings. We mapped 1-2 stars as negative (0) and 4-5 stars as positive (1) to create a binary classification target, discarding ambiguous 3-star reviews.

## Approach
1. **Data Cleaning & EDA** — Filtered and mapped multi-class star ratings into binary sentiment labels, performing basic text sanitization.
2. **Feature Engineering** — Applied TF-IDF vectorization (max 5000 features) for heuristic models, and leveraged deeply contextual embeddings via HuggingFace's `AutoTokenizer` for transformer models.
3. **Modeling** — Evaluated traditional heuristic classifiers (Logistic Regression, Multinomial Naive Bayes) against a state-of-the-art contextual transformer (`distilbert-base-uncased`). Applied unsupervised Latent Dirichlet Allocation (LDA) on the negative predictions to extract topics.
4. **Evaluation** — Tracked macro-averaged F1, Precision, and Recall alongside raw accuracy to ensure minority classes (negative reviews) were appropriately weighted in the evaluation.

## Key Results
| Metric | TF-IDF + LR | Multinomial NB | DistilBERT |
|--------|-------|-------|-------|
| F1-Score (Macro) | 0.72 | 0.61 | 0.87 |
| Precision | 0.76 | 0.65 | 0.87 |
| Accuracy | 0.83 | 0.82 | 0.87 |

## Key Findings
1. DistilBERT achieved an F1-score of 0.87, significantly outperforming heuristic baselines (TF-IDF + LR at 0.72) in handling contextual nuances like sarcasm and negation.
2. Unsupervised LDA topic modeling on negative reviews revealed 5 distinct complaint clusters, prominently exposing hardware and physical defects (e.g., connectors, switches, materials).
3. **Business Implication:** Deploying a transformer-based classifier paired with LDA topic extraction allows businesses to automatically route unstructured product complaints directly to engineering or QA teams, reducing the need for manual triage.

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the baseline and transformer pipelines
python src/baseline_models.py
python src/transformer_finetune.py

# 3. Extract topics and generate the PDF report
python src/topic_modeling.py
python src/report_generator.py
```

## Files
- `src/` — Production-ready scripts containing data loaders, modeling pipelines, and the reporting engine.
- `report.pdf` — The final generated IEEE-style academic evaluation report.
- `baseline_metrics.json` / `transformer_metrics.json` — Raw evaluation output data.

## Methodology & Limitations
This project employs a comparative methodology, benchmarking computationally inexpensive bag-of-words models against modern attention-based transformers. By tracking precision, recall, and F1 alongside raw accuracy, the evaluation mitigates the deceptive metrics often found in imbalanced organic review data. The secondary phase utilizes unsupervised Latent Dirichlet Allocation (LDA) to group the negative predictions into actionable defect categories.

**Limitations:** The dataset was heavily subsampled to 10,000 records to accommodate local training constraints. Consequently, the model's vocabulary and contextual understanding are constrained to the domains present in this subset. Additionally, the LDA topic extraction relies on a static 5-cluster assumption; in a true production environment, the optimal number of topics (`k`) would need to be determined dynamically using coherence scores. Finally, the pipeline currently lacks a live real-time inference endpoint, functioning instead as a batch reporting engine.
