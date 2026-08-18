# Project Guardrails

The following guardrails must be strictly adhered to during development on `nlp_sentiment_analysis`:

## 1. Security First
- **No Unsafe Serialization**: Never use `pickle` for model metrics, configuration, or state management. Use `json`, `safetensors`, or `joblib` strictly within bounded contexts.
- **Dependency Audits**: Run `pip-audit` or `safety` checks before any production deployment.
- **Input Validation**: All data loading and string parsing must be validated (e.g., using `pydantic` models as seen in `DataConfig`).

## 2. Code Quality
- **Typing**: All functions must have Python type hints (`typing.Dict`, `typing.Any`, etc.).
- **Professional Language**: Do not use casual, "AI-ish" generated phrases (e.g., "delve into", "testament to", "let's dive in") in documentation or user-facing reports.

## 3. Operations
- **Reproducibility**: Always enforce a global random seed for `load_dataset` and `train_test_split`.
- **Metrics Integrity**: The `report_generator.py` must dynamically pull metrics from JSON artifacts; hardcoded metric tracking is prohibited.
