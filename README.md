# E-Commerce Sentiment Analysis & Triage Pipeline

## Problem Statement
Can customer sentiment be accurately classified from unstructured review text at scale, and what latent themes drive negative feedback? 

This project aims to automate the triage of e-commerce product reviews by classifying sentiment and extracting actionable operational intelligence (e.g., hardware defects, shipping delays) from negative reviews.

## Skills Demonstrated
- **Natural Language Processing (NLP):** Tokenization, TF-IDF vectorization, Contextual embeddings.
- **Machine Learning:** Logistic Regression, Multinomial Naive Bayes, DistilBERT (Transformer fine-tuning via HuggingFace).
- **Unsupervised Learning:** Latent Dirichlet Allocation (LDA) for topic modeling.
- **MLOps & Automation:** Automated PDF report generation (ReportLab), CI/CD pipelines (GitHub Actions), Security scanning (Bandit).

## Deliverables
- **Data Pipeline:** Automated ingestion and preprocessing using the HuggingFace `datasets` library.
- **Classification Engine:** Comparative evaluation of heuristic models (TF-IDF + LR/NB) against a state-of-the-art transformer (`distilbert-base-uncased`).
- **Topic Extractor:** LDA implementation targeting negative sentiment to cluster root causes.
- **Automated Reporting:** An IEEE-formatted, academic-style PDF report summarizing system performance and business findings.

## Methodology & Architecture
1. **Data Ingestion:** Loads a subset of product reviews directly from the HuggingFace Hub.
2. **Baseline Modeling:** Implements traditional TF-IDF vectorization (max_features=5000) coupled with Logistic Regression and Multinomial Naive Bayes.
3. **Transformer Fine-Tuning:** Leverages PyTorch and the HuggingFace `Trainer` API to fine-tune DistilBERT, accommodating class imbalances by tracking macro-averaged F1, Precision, and Recall.
4. **Topic Modeling:** Filters for negative reviews and applies a 5-component LDA model to extract dominant complaint themes.
5. **Reporting:** Aggregates JSON metrics into a reproducible, two-column academic PDF using ReportLab and Seaborn.

## Key Results
| Model | F1-Score | Accuracy |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 0.72 | 0.83 |
| Multinomial Naive Bayes | 0.61 | 0.82 |
| DistilBERT | **0.87** | **0.87** |

*Note: Results may vary slightly depending on the training seed and data splits.*

## Key Findings
1. **Contextual Superiority:** The DistilBERT architecture significantly outperforms heuristic baselines, particularly in resolving complex linguistic structures like sarcasm and negation that deceive bag-of-words models.
2. **Hardware Defect Identification:** LDA topic extraction on negative reviews successfully clustered specific physical failure domains (e.g., connectors, switches), transforming unstructured complaints into actionable engineering alerts.
3. **Operational Implication:** Relying strictly on accuracy metrics is deceptive in imbalanced datasets. Macro-averaged F1-scores confirm the transformer's superior ability to identify minority negative classes.

## How to Run

Ensure you have a Python 3.10+ environment.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the baseline models
python src/baseline_models.py

# 3. Fine-tune the transformer (Requires GPU for optimal performance)
python src/transformer_finetune.py

# 4. Run topic modeling on negative reviews
python src/topic_modeling.py

# 5. Generate the final IEEE-style PDF report
python src/report_generator.py
```

## Files
- `src/data_loader.py` - HuggingFace dataset fetching and preparation.
- `src/baseline_models.py` - TF-IDF and heuristic classifiers.
- `src/transformer_finetune.py` - DistilBERT fine-tuning logic.
- `src/topic_modeling.py` - LDA extraction on negative sentiment.
- `src/report_generator.py` - Automated PDF generation.
- `report.pdf` - The final generated academic output.
