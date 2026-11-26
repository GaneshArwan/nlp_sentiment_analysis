# E-Commerce Sentiment Analysis & Triage Pipeline

## Problem Statement
Can customer sentiment be accurately classified from unstructured review text at scale, and what latent themes drive negative feedback?

## Dataset
- **Source:** HuggingFace Hub (`fancyzhx/amazon_polarity`)
- **Size:** Sampled subset of 2,000 training reviews and 400 test reviews (configurable via `DataConfig.sample_size`, seed `42`)
- **Description:** Binary Amazon product reviews (`0` = negative, `1` = positive) with HTML tag stripping and text normalization in `src/data_loader.py`.

## Approach
1. **Data Cleaning & Subsampling** — Loaded `fancyzhx/amazon_polarity` with deterministic seed `42` and cleaned review text (HTML removal, punctuation normalization, lowercasing).
2. **Feature Engineering** — Applied TF-IDF vectorization (max 5000 features) for heuristic models, and contextual tokenization (`AutoTokenizer`, `max_length=128`) for the transformer model.
3. **Modeling** — Evaluated heuristic classifiers (Logistic Regression, Multinomial Naive Bayes) against a fine-tuned transformer (`distilbert-base-uncased`). Applied unsupervised Latent Dirichlet Allocation (LDA, `k=5`) on negative training reviews to extract complaint topics.
4. **Evaluation & Dashboard** — Tracked macro-averaged F1, Precision, Recall, and Accuracy with 95% confidence intervals, rendered into a 3-page Data Science & Model QA PDF report and an interactive Streamlit dashboard.

## Key Results
Evaluated on `N=400` test samples (`baseline_metrics.json` and `transformer_metrics.json`):

| Metric | TF-IDF + LR | Multinomial NB | DistilBERT |
|--------|-------|-------|-------|
| F1-Score (Macro) | 0.812 | 0.792 | 0.895 |
| Precision (Macro) | 0.813 | 0.793 | 0.895 |
| Recall (Macro) | 0.813 | 0.793 | 0.895 |
| Accuracy | 0.813 | 0.793 | 0.895 |

![Model Evaluation Comparison](accuracy_chart.png)

## Key Findings
1. DistilBERT achieved a macro F1-score of `0.895` (`89.5%` accuracy), outperforming TF-IDF + Logistic Regression (`0.812`) and Multinomial Naive Bayes (`0.792`).
2. Unsupervised LDA topic modeling on negative reviews surfaced 5 complaint clusters spanning games/media, music CDs, books/movies, general product quality/price, and hardware/connector issues.
3. **Business Implication:** Pairing a transformer sentiment classifier with LDA topic extraction enables automated routing of negative product feedback to engineering or QA teams.

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the baseline (+ LDA topic modeling) and transformer pipelines
python src/baseline_models.py
python src/transformer_finetune.py

# 3. Generate charts and the PDF report
python src/report_generator.py

# 4. Launch the interactive Streamlit dashboard
streamlit run app.py

# 5. Run the unit test suite
pytest -v
```

### Docker
```bash
docker build -t nlp-sentiment-dashboard .
docker run -p 8501:8501 nlp-sentiment-dashboard
```

## Files
- `src/` — Data loader, baseline/LDA training, transformer fine-tuning, and PDF report generator.
- `app.py` — Streamlit dashboard displaying model metrics, 95% CIs, charts, and LDA topics.
- `tests/` — Offline unit tests (`pytest`).
- `report.pdf` — 3-page Data Science & Model QA evaluation report.
- `accuracy_chart.png` / `class_metrics_chart.png` — High-resolution evaluation visualizations.
- `baseline_metrics.json` / `transformer_metrics.json` / `lda_topics.json` — Raw evaluation outputs.

## Methodology & Limitations
This project benchmarks computationally inexpensive bag-of-words baselines against a fine-tuned attention-based transformer (`distilbert-base-uncased`). Tracking macro-averaged precision, recall, and F1 alongside accuracy provides balanced evaluation across sentiment classes, while LDA groups negative reviews into interpretable themes.

**Limitations:** Default `DataConfig` subsamples 2,000 training and 400 test records to accommodate local CPU/GPU training constraints. Additionally, LDA topic extraction uses a fixed 5-cluster configuration; production deployments should tune cluster count dynamically via coherence evaluation and add a dedicated real-time inference API endpoint beyond the batch pipeline and reporting dashboard.
