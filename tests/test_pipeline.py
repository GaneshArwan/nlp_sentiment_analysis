"""Offline deterministic unit tests for the NLP sentiment analysis pipeline."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from unittest.mock import MagicMock

import matplotlib
import pytest
from pydantic import ValidationError
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import baseline_models  # noqa: E402
import data_loader  # noqa: E402
import report_generator  # noqa: E402
import transformer_finetune  # noqa: E402


class _SyntheticTrainSplit:
    """Minimal deterministic dataset split supporting .filter() for LDA tests."""

    def __init__(self, rows: list[Dict[str, Any]]) -> None:
        self._rows = rows

    def __getitem__(self, key: str) -> list[Any]:
        return [row[key] for row in self._rows]

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "_SyntheticTrainSplit":
        return _SyntheticTrainSplit([row for row in self._rows if predicate(row)])


@pytest.fixture
def synthetic_sentiment_data() -> Dict[str, Dict[str, list[Any]]]:
    """Provide deterministic synthetic train/test splits for baseline sentiment models."""
    return {
        "train": {
            "cleaned_text": [
                "excellent product fast delivery works great",
                "amazing quality love design highly recommend",
                "wonderful purchase battery lasts long time",
                "perfect item arrived early very happy",
                "terrible product broke immediately total waste",
                "awful quality stopped working cheap plastic",
                "horrible experience damaged box missing parts",
                "worst purchase disappointed refund requested immediately",
            ],
            "label": [1, 1, 1, 1, 0, 0, 0, 0],
        },
        "test": {
            "cleaned_text": [
                "great product love quality",
                "happy purchase works fast",
                "terrible broken cheap waste",
                "horrible damaged disappointed",
            ],
            "label": [1, 1, 0, 0],
        },
    }


@pytest.fixture
def synthetic_topic_dataset() -> Dict[str, _SyntheticTrainSplit]:
    """Provide deterministic synthetic dataset with .filter() for LDA topic modeling."""
    rows = [
        {"cleaned_text": "excellent battery life fast shipping", "label": 1},
        {"cleaned_text": "battery died screen broken refund", "label": 0},
        {"cleaned_text": "broken screen plastic cheap refund", "label": 0},
        {"cleaned_text": "shipping delayed damaged package refund", "label": 0},
        {"cleaned_text": "damaged box missing cable shipping", "label": 0},
        {"cleaned_text": "battery overheated plastic cheap broken", "label": 0},
        {"cleaned_text": "delayed delivery damaged screen missing", "label": 0},
    ]
    return {"train": _SyntheticTrainSplit(rows)}


@pytest.fixture
def valid_report_artifacts(tmp_path: Path) -> Dict[str, Dict[str, Any]]:
    """Create valid synthetic JSON metric artifacts inside tmp_path."""
    baseline_metrics = {
        "n_test_samples": 20,
        "lr_accuracy": 0.85,
        "lr_report": {
            "0": {"support": 10},
            "1": {"support": 10},
            "macro avg": {
                "precision": 0.85,
                "recall": 0.85,
                "f1-score": 0.85,
                "support": 20,
            },
        },
        "nb_accuracy": 0.80,
        "nb_report": {
            "0": {"support": 10},
            "1": {"support": 10},
            "macro avg": {
                "precision": 0.80,
                "recall": 0.80,
                "f1-score": 0.80,
                "support": 20,
            },
        },
    }
    lda_topics = {
        "0": "battery screen broken plastic cheap",
        "1": "shipping delayed damaged missing refund",
    }
    transformer_metrics = {
        "eval_accuracy": 0.92,
        "eval_precision": 0.92,
        "eval_recall": 0.92,
        "eval_f1": 0.92,
    }

    (tmp_path / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics), encoding="utf-8"
    )
    (tmp_path / "lda_topics.json").write_text(
        json.dumps(lda_topics), encoding="utf-8"
    )
    (tmp_path / "transformer_metrics.json").write_text(
        json.dumps(transformer_metrics), encoding="utf-8"
    )
    return {
        "baseline_metrics": baseline_metrics,
        "lda_topics": lda_topics,
        "transformer_metrics": transformer_metrics,
    }


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import app.py with a mocked streamlit module so st.set_page_config runs safely."""
    mock_st = MagicMock()
    monkeypatch.setitem(sys.modules, "streamlit", mock_st)
    if "app" in sys.modules:
        return importlib.reload(sys.modules["app"])
    return importlib.import_module("app")


class TestDataLoader:
    """Unit tests for text cleaning and configuration in src/data_loader.py."""

    @pytest.mark.parametrize(
        ("raw_text", "expected"),
        [
            ("<p>Hello <b>World</b>!</p>", "hello world"),
            ("Order #12345 cost $99.99 & failed!!!", "order cost failed"),
            ("  Too   many\t\nspaces   here  ", "too many spaces here"),
            ("<div class='x'>Item 42</div>   rocks!!", "item rocks"),
            ("", ""),
        ],
    )
    def test_clean_text_valid_strings_normalized_correctly(
        self, raw_text: str, expected: str
    ) -> None:
        """Verify HTML tags, digits, punctuation, case, and whitespace are normalized."""
        # Arrange
        input_value = raw_text

        # Act
        result = data_loader.clean_text(input_value)

        # Assert
        assert result == expected

    @pytest.mark.parametrize("non_string", [None, 123, 45.67, ["review"], {"text": "hi"}])
    def test_clean_text_non_string_input_returns_empty_string(
        self, non_string: Any
    ) -> None:
        """Verify non-string inputs safely return an empty string."""
        # Arrange
        input_value = non_string

        # Act
        result = data_loader.clean_text(input_value)

        # Assert
        assert result == ""

    def test_data_config_defaults_match_expected_values(self) -> None:
        """Verify DataConfig initializes with expected default field values."""
        # Arrange
        expected_dataset = "fancyzhx/amazon_polarity"

        # Act
        config = data_loader.DataConfig()

        # Assert
        assert config.dataset_name == expected_dataset
        assert config.sample_size == 2000
        assert config.seed == 42

    def test_data_config_invalid_types_raises_validation_error(self) -> None:
        """Verify DataConfig rejects non-coercible field types via Pydantic validation."""
        # Arrange
        invalid_kwargs = {"sample_size": "not-an-int", "seed": "bad-seed"}

        # Act / Assert
        with pytest.raises(ValidationError):
            data_loader.DataConfig(**invalid_kwargs)


class TestBaselineModels:
    """Unit tests for baseline training and artifact persistence in src/baseline_models.py."""

    def test_baseline_config_defaults_match_expected_values(self) -> None:
        """Verify BaselineConfig initializes with expected default hyperparameters."""
        # Arrange
        expected_features = 5000

        # Act
        config = baseline_models.BaselineConfig()

        # Assert
        assert config.max_features == expected_features
        assert config.lda_n_components == 5
        assert config.random_state == 42

    def test_train_sentiment_models_synthetic_data_returns_expected_metrics(
        self, synthetic_sentiment_data: Dict[str, Dict[str, list[Any]]]
    ) -> None:
        """Verify sentiment models return valid metrics, n_test_samples, and vectorizer."""
        # Arrange
        config = baseline_models.BaselineConfig(max_features=100, random_state=42)
        expected_test_len = len(synthetic_sentiment_data["test"]["label"])

        # Act
        models, vectorizer, metrics = baseline_models.train_sentiment_models(
            synthetic_sentiment_data, config
        )

        # Assert
        assert set(models.keys()) == {"logistic_regression", "naive_bayes"}
        assert isinstance(vectorizer, TfidfVectorizer)
        assert metrics["n_test_samples"] == expected_test_len
        assert 0.0 <= metrics["lr_accuracy"] <= 1.0
        assert 0.0 <= metrics["nb_accuracy"] <= 1.0
        assert "macro avg" in metrics["lr_report"]
        assert "macro avg" in metrics["nb_report"]

    def test_train_topic_model_synthetic_dataset_extracts_topics(
        self, synthetic_topic_dataset: Dict[str, _SyntheticTrainSplit]
    ) -> None:
        """Verify LDA topic model filters negative reviews and extracts topic terms."""
        # Arrange
        config = baseline_models.BaselineConfig(
            max_features=50, lda_n_components=2, random_state=42
        )

        # Act
        lda, tf_vectorizer, topics = baseline_models.train_topic_model(
            synthetic_topic_dataset, config
        )

        # Assert
        assert isinstance(lda, LatentDirichletAllocation)
        assert isinstance(tf_vectorizer, CountVectorizer)
        assert list(topics.keys()) == [0, 1]
        assert all(isinstance(terms, str) and len(terms.split()) > 0 for terms in topics.values())

    def test_save_baseline_artifacts_valid_dicts_writes_json_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify save_baseline_artifacts persists baseline_metrics.json and lda_topics.json."""
        # Arrange
        monkeypatch.chdir(tmp_path)
        metrics = {"n_test_samples": 4, "lr_accuracy": 0.75}
        topics = {0: "battery screen broken"}

        # Act
        baseline_models.save_baseline_artifacts(metrics, topics)

        # Assert
        metrics_file = tmp_path / "baseline_metrics.json"
        topics_file = tmp_path / "lda_topics.json"
        assert metrics_file.exists()
        assert topics_file.exists()
        assert json.loads(metrics_file.read_text(encoding="utf-8")) == metrics
        assert json.loads(topics_file.read_text(encoding="utf-8")) == {"0": "battery screen broken"}


class TestReportGenerator:
    """Unit tests for confidence intervals, JSON loading, and PDF generation."""

    @pytest.mark.parametrize(
        ("p", "n", "expected"),
        [
            (0.5, 100, 1.96 * math.sqrt(0.25 / 100)),
            (0.0, 100, 0.0),
            (1.0, 100, 0.0),
            (-0.5, 100, 0.0),
            (1.5, 100, 0.0),
            (0.5, 0, 1.96 * math.sqrt(0.25 / 1)),
            (0.5, -10, 1.96 * math.sqrt(0.25 / 1)),
        ],
    )
    def test_calc_accuracy_ci_edge_and_standard_cases_returns_clamped_ci(
        self, p: float, n: int, expected: float
    ) -> None:
        """Verify calc_accuracy_ci computes binomial CI and clamps invalid p or n."""
        # Arrange
        proportion = p
        sample_size = n

        # Act
        result = report_generator.calc_accuracy_ci(proportion, sample_size)

        # Assert
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("p", "n", "expected"),
        [
            (0.5, 100, 1.96 * math.sqrt(1.5 * 0.25 / 100)),
            (0.0, 50, 0.0),
            (1.0, 50, 0.0),
            (-0.2, 50, 0.0),
            (1.2, 50, 0.0),
            (0.5, 0, 1.96 * math.sqrt(1.5 * 0.25 / 1)),
        ],
    )
    def test_approx_composite_ci_edge_and_standard_cases_returns_clamped_vif_ci(
        self, p: float, n: int, expected: float
    ) -> None:
        """Verify approx_composite_ci applies VIF=1.5 and guards against n<=0 and out-of-range p."""
        # Arrange
        score = p
        sample_size = n

        # Act
        result = report_generator.approx_composite_ci(score, sample_size)

        # Assert
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("metric_name", "expected_fn"),
        [
            ("Accuracy", report_generator.calc_accuracy_ci),
            ("Precision", report_generator.approx_composite_ci),
            ("Recall", report_generator.approx_composite_ci),
            ("F1-Score", report_generator.approx_composite_ci),
        ],
    )
    def test_get_ci_for_metric_dispatches_expected_ci_formula(
        self, metric_name: str, expected_fn: Callable[[float, int], float]
    ) -> None:
        """Verify get_ci_for_metric routes Accuracy to binomial CI and others to composite CI."""
        # Arrange
        score = 0.8
        n_test = 100

        # Act
        result = report_generator.get_ci_for_metric(metric_name, score, n_test)

        # Assert
        assert result == pytest.approx(expected_fn(score, n_test))

    def test_load_and_validate_metrics_valid_json_dict_returns_dict(
        self, tmp_path: Path
    ) -> None:
        """Verify load_and_validate_metrics loads and returns a valid JSON dictionary."""
        # Arrange
        payload = {"eval_accuracy": 0.9, "eval_f1": 0.89}
        file_path = tmp_path / "metrics.json"
        file_path.write_text(json.dumps(payload), encoding="utf-8")

        # Act
        result = report_generator.load_and_validate_metrics(str(file_path))

        # Assert
        assert result == payload

    def test_load_and_validate_metrics_missing_file_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Verify load_and_validate_metrics raises ValueError when file does not exist."""
        # Arrange
        missing_path = tmp_path / "missing.json"

        # Act / Assert
        with pytest.raises(ValueError, match="Error loading"):
            report_generator.load_and_validate_metrics(str(missing_path))

    def test_load_and_validate_metrics_invalid_json_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Verify load_and_validate_metrics raises ValueError on malformed JSON."""
        # Arrange
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{bad json", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="Error loading"):
            report_generator.load_and_validate_metrics(str(bad_file))

    def test_load_and_validate_metrics_non_dict_json_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Verify load_and_validate_metrics raises ValueError when JSON root is not a dict."""
        # Arrange
        list_file = tmp_path / "list.json"
        list_file.write_text("[1, 2, 3]", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ValueError, match="Expected JSON object"):
            report_generator.load_and_validate_metrics(str(list_file))

    def test_build_pdf_report_valid_json_creates_chart_and_pdf(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        valid_report_artifacts: Dict[str, Dict[str, Any]],
    ) -> None:
        """Verify build_pdf_report writes non-empty accuracy_chart.png and report.pdf."""
        # Arrange
        monkeypatch.chdir(tmp_path)
        assert "baseline_metrics" in valid_report_artifacts

        # Act
        report_generator.build_pdf_report()

        # Assert
        chart_path = tmp_path / "accuracy_chart.png"
        pdf_path = tmp_path / "report.pdf"
        assert chart_path.exists() and chart_path.stat().st_size > 0
        assert pdf_path.exists() and pdf_path.stat().st_size > 0


class TestAppDashboardHelpers:
    """Unit tests for Streamlit-safe helper functions in app.py."""

    @pytest.mark.parametrize(
        ("p", "n", "expected"),
        [
            (0.5, 100, 1.96 * math.sqrt(0.25 / 100)),
            (0.0, 100, 0.0),
            (1.0, 100, 0.0),
            (-0.1, 100, 0.0),
            (1.1, 100, 0.0),
            (0.5, 0, 1.96 * math.sqrt(0.25 / 1)),
        ],
    )
    def test_calc_ci_edge_and_standard_cases_returns_clamped_ci(
        self, app_module: Any, p: float, n: int, expected: float
    ) -> None:
        """Verify app.calc_ci computes binomial CI with n=0 guard and p clamping."""
        # Arrange
        proportion = p
        sample_size = n

        # Act
        result = app_module.calc_ci(proportion, sample_size)

        # Assert
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("p", "n", "expected"),
        [
            (0.5, 100, 1.96 * math.sqrt(1.5 * 0.25 / 100)),
            (0.0, 100, 0.0),
            (1.0, 100, 0.0),
            (-0.5, 100, 0.0),
            (1.5, 100, 0.0),
            (0.5, 0, 1.96 * math.sqrt(1.5 * 0.25 / 1)),
        ],
    )
    def test_calc_composite_ci_edge_and_standard_cases_returns_clamped_vif_ci(
        self, app_module: Any, p: float, n: int, expected: float
    ) -> None:
        """Verify app.calc_composite_ci applies VIF=1.5 with n=0 guard and p clamping."""
        # Arrange
        score = p
        sample_size = n

        # Act
        result = app_module.calc_composite_ci(score, sample_size)

        # Assert
        assert result == pytest.approx(expected)

    def test_load_data_valid_json_returns_parsed_dict(
        self, app_module: Any, tmp_path: Path
    ) -> None:
        """Verify app.load_data returns parsed JSON dictionary for valid files."""
        # Arrange
        payload = {"eval_accuracy": 0.91}
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(json.dumps(payload), encoding="utf-8")

        # Act
        result = app_module.load_data(str(valid_file))

        # Assert
        assert result == payload

    def test_load_data_missing_or_corrupt_file_returns_none(
        self, app_module: Any, tmp_path: Path
    ) -> None:
        """Verify app.load_data returns None when file is missing, corrupt, or non-dict JSON."""
        # Arrange
        missing_file = tmp_path / "does_not_exist.json"
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{corrupt", encoding="utf-8")
        list_file = tmp_path / "list.json"
        list_file.write_text("[1, 2, 3]", encoding="utf-8")

        # Act
        missing_result = app_module.load_data(str(missing_file))
        corrupt_result = app_module.load_data(str(corrupt_file))
        list_result = app_module.load_data(str(list_file))

        # Assert
        assert missing_result is None
        assert corrupt_result is None
        assert list_result is None

    def test_calc_ci_nan_input_returns_zero(self, app_module: Any) -> None:
        """Verify app.calc_ci and calc_composite_ci return 0.0 for NaN inputs."""
        # Arrange
        nan_value = float("nan")

        # Act
        ci_result = app_module.calc_ci(nan_value, 100)
        comp_result = app_module.calc_composite_ci(nan_value, 100)

        # Assert
        assert ci_result == 0.0
        assert comp_result == 0.0


class TestConfigValidationAndReportBounds:
    """Unit tests for Pydantic bounds, TransformerConfig, and report_generator guards."""

    def test_transformer_config_defaults_and_bounds_validated(self) -> None:
        """Verify TransformerConfig sets reproducible defaults and rejects invalid bounds."""
        # Arrange / Act
        config = transformer_finetune.TransformerConfig()

        # Assert
        assert config.model_name == "distilbert-base-uncased"
        assert config.epochs == 1
        assert config.batch_size == 8
        assert config.seed == 42

        with pytest.raises(ValidationError):
            transformer_finetune.TransformerConfig(epochs=0)
        with pytest.raises(ValidationError):
            transformer_finetune.TransformerConfig(batch_size=0)
        with pytest.raises(ValidationError):
            transformer_finetune.TransformerConfig(seed=-1)

    def test_data_and_baseline_configs_reject_invalid_bounds(self) -> None:
        """Verify DataConfig and BaselineConfig enforce positive sizes and non-negative seeds."""
        # Arrange / Act / Assert
        with pytest.raises(ValidationError):
            data_loader.DataConfig(sample_size=4)
        with pytest.raises(ValidationError):
            data_loader.DataConfig(seed=-1)
        with pytest.raises(ValidationError):
            baseline_models.BaselineConfig(max_features=0)
        with pytest.raises(ValidationError):
            baseline_models.BaselineConfig(lda_n_components=0)

    def test_build_pdf_report_escapes_xml_and_rejects_bool_metric(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        valid_report_artifacts: Dict[str, Dict[str, Any]],
    ) -> None:
        """Verify XML special characters in topics succeed and bool metrics raise TypeError."""
        # Arrange
        monkeypatch.chdir(tmp_path)
        (tmp_path / "lda_topics.json").write_text(
            json.dumps({"0": "cable <broken> & cheap > price"}), encoding="utf-8"
        )

        # Act
        report_generator.build_pdf_report()

        # Assert
        assert (tmp_path / "report.pdf").exists()

        bad_transformer = dict(valid_report_artifacts["transformer_metrics"])
        bad_transformer["eval_accuracy"] = True
        (tmp_path / "transformer_metrics.json").write_text(
            json.dumps(bad_transformer), encoding="utf-8"
        )
        with pytest.raises(TypeError, match="not a numeric type"):
            report_generator.build_pdf_report()

    @pytest.mark.parametrize(
        ("p1", "p2", "n1", "n2", "expected_z_sign"),
        [
            (0.895, 0.8125, 400, 400, 1),
            (0.80, 0.80, 400, 400, 0),
            (0.75, 0.85, 200, 200, -1),
            (1.0, 1.0, 100, 100, 0),
            (float("nan"), 0.80, 100, 100, 0),
        ],
    )
    def test_calc_two_proportion_ztest_returns_expected_z_and_pvalue(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int,
        expected_z_sign: int,
    ) -> None:
        """Verify two-proportion z-test handles lift, equal/degenerate proportions, and NaN."""
        # Arrange / Act
        z_stat, p_val = report_generator.calc_two_proportion_ztest(p1, p2, n1, n2)

        # Assert
        assert 0.0 <= p_val <= 1.0
        if expected_z_sign == 0:
            assert z_stat == 0.0
            assert p_val == 1.0
        elif expected_z_sign > 0:
            assert z_stat > 0.0
            assert p_val < 0.01
        else:
            assert z_stat < 0.0
            assert p_val < 0.05

    def test_build_pdf_report_with_full_per_class_metrics_creates_both_charts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        valid_report_artifacts: Dict[str, Dict[str, Any]],
    ) -> None:
        """Verify build_pdf_report generates class_metrics_chart.png when per-class stats exist."""
        # Arrange
        monkeypatch.chdir(tmp_path)
        baseline = dict(valid_report_artifacts["baseline_metrics"])
        for rep_key in ("lr_report", "nb_report"):
            baseline[rep_key]["0"] = {"precision": 0.80, "recall": 0.82, "f1-score": 0.81, "support": 10}
            baseline[rep_key]["1"] = {"precision": 0.84, "recall": 0.82, "f1-score": 0.83, "support": 10}
        (tmp_path / "baseline_metrics.json").write_text(
            json.dumps(baseline), encoding="utf-8"
        )

        # Act
        report_generator.build_pdf_report()

        # Assert
        assert (tmp_path / "accuracy_chart.png").exists()
        assert (tmp_path / "class_metrics_chart.png").exists()
        assert (tmp_path / "report.pdf").exists()
