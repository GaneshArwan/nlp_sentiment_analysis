"""Publication-grade PDF evaluation report generator for NLP sentiment models."""

from html import escape
import json
import math
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
import seaborn as sns

# Palette
NAVY = colors.HexColor("#1B365D")
ACCENT_BLUE = colors.HexColor("#2B6CB0")
SLATE = colors.HexColor("#4A5568")
MUTED = colors.HexColor("#718096")
LIGHT_BG = colors.HexColor("#F4F6F9")
ALT_ROW = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#CBD5E0")
DARK_TEXT = colors.HexColor("#1A202C")


def _is_valid_number(val: Any) -> bool:
    """Return True if val is a finite int/float (excluding bool)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def calc_accuracy_ci(p: float, n: int) -> float:
    """Calculate 95% confidence interval half-width for a binomial proportion."""
    if not _is_valid_number(p):
        return 0.0
    p_clamped = min(max(float(p), 0.0), 1.0)
    n_safe = max(int(n), 1)
    return float(1.96 * np.sqrt((p_clamped * (1.0 - p_clamped)) / n_safe))


def approx_composite_ci(p: float, n: int) -> float:
    """
    Approximate 95% CI half-width for composite metrics (Precision, Recall, F1).

    Uses a conservative variance inflation factor (VIF = 1.5) on binomial
    variance because paired per-sample predictions are not persisted.
    """
    if not _is_valid_number(p):
        return 0.0
    p_clamped = min(max(float(p), 0.0), 1.0)
    n_safe = max(int(n), 1)
    vif = 1.5
    return float(1.96 * np.sqrt(vif * (p_clamped * (1.0 - p_clamped)) / n_safe))


def get_ci_for_metric(metric_name: str, score: float, n_test: int) -> float:
    """Dispatch exact binomial CI for Accuracy and VIF-inflated CI otherwise."""
    if metric_name == "Accuracy":
        return calc_accuracy_ci(score, n_test)
    return approx_composite_ci(score, n_test)


def calc_two_proportion_ztest(p1: float, p2: float, n1: int, n2: int) -> Tuple[float, float]:
    """
    Compute an independent two-proportion z-statistic and two-sided p-value.

    Used as a large-sample approximation when paired predictions (for McNemar's
    test) are unavailable in summary JSON artifacts.
    """
    if not (_is_valid_number(p1) and _is_valid_number(p2)):
        return 0.0, 1.0
    p1_c = min(max(float(p1), 0.0), 1.0)
    p2_c = min(max(float(p2), 0.0), 1.0)
    n1_safe = max(int(n1), 1)
    n2_safe = max(int(n2), 1)

    p_pool = (p1_c * n1_safe + p2_c * n2_safe) / (n1_safe + n2_safe)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1_safe + 1.0 / n2_safe))
    if se == 0.0:
        return 0.0, 1.0
    z_stat = (p1_c - p2_c) / se
    p_val = math.erfc(abs(z_stat) / math.sqrt(2.0))
    return float(z_stat), float(p_val)


def load_and_validate_metrics(filepath: str) -> Dict[str, Any]:
    """Load a JSON object artifact from disk and validate its root type."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {filepath}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading {filepath}: {str(e)}")


def _has_per_class_metrics(baseline_metrics: Dict[str, Any]) -> bool:
    """Return True when lr_report and nb_report contain per-class P/R/F1."""
    required_keys = {"precision", "recall", "f1-score", "support"}
    for report_key in ("lr_report", "nb_report"):
        report = baseline_metrics.get(report_key, {})
        for cls_key in ("0", "1"):
            cls_metrics = report.get(cls_key, {})
            if not isinstance(cls_metrics, dict) or not required_keys.issubset(cls_metrics.keys()):
                return False
            if not all(_is_valid_number(cls_metrics[k]) for k in required_keys):
                return False
    return True


def generate_charts(
    baseline_metrics: Dict[str, Any],
    transformer_metrics: Dict[str, Any],
    n_test: int,
) -> None:
    """Generate Fig. 1 (overall model comparison) and Fig. 2 (per-class baselines)."""
    lr_acc = float(baseline_metrics["lr_accuracy"])
    lr_p = float(baseline_metrics["lr_report"]["macro avg"]["precision"])
    lr_r = float(baseline_metrics["lr_report"]["macro avg"]["recall"])
    lr_f1 = float(baseline_metrics["lr_report"]["macro avg"]["f1-score"])

    nb_acc = float(baseline_metrics["nb_accuracy"])
    nb_p = float(baseline_metrics["nb_report"]["macro avg"]["precision"])
    nb_r = float(baseline_metrics["nb_report"]["macro avg"]["recall"])
    nb_f1 = float(baseline_metrics["nb_report"]["macro avg"]["f1-score"])

    t_acc = float(transformer_metrics["eval_accuracy"])
    t_p = float(transformer_metrics["eval_precision"])
    t_r = float(transformer_metrics["eval_recall"])
    t_f1 = float(transformer_metrics["eval_f1"])

    rows = []
    models = ["TF-IDF + LR", "Multinomial NB", "DistilBERT"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    all_scores = [lr_acc, lr_p, lr_r, lr_f1, nb_acc, nb_p, nb_r, nb_f1, t_acc, t_p, t_r, t_f1]

    for model, scores in zip(
        models,
        [[lr_acc, lr_p, lr_r, lr_f1], [nb_acc, nb_p, nb_r, nb_f1], [t_acc, t_p, t_r, t_f1]],
    ):
        for metric, score in zip(metrics, scores):
            ci = get_ci_for_metric(metric, score, n_test)
            rows.append({"Model": model, "Metric": metric, "Score": score, "CI": ci})

    df = pd.DataFrame(rows)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(models))
    width = 0.19
    palette = sns.color_palette("colorblind", len(metrics))

    for i, metric in enumerate(metrics):
        sub = df[df["Metric"] == metric]
        scores_arr = [float(sub[sub["Model"] == m]["Score"].iloc[0]) for m in models]
        ci_arr = [float(sub[sub["Model"] == m]["CI"].iloc[0]) for m in models]
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset,
            scores_arr,
            width,
            label=metric,
            yerr=ci_arr,
            capsize=3,
            color=palette[i],
            edgecolor="#2D3748",
            linewidth=0.5,
            error_kw={"elinewidth": 1.0, "ecolor": "#2D3748"},
        )
        for bar, score, err in zip(bars, scores_arr, ci_arr):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(score + err + 0.012, 1.03),
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#1A202C",
            )

    min_score = min(all_scores)
    y_min = 0.50 if min_score >= 0.60 else 0.0
    ax.set_ylim(y_min, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, fontweight="bold", color="#1A202C")
    ax.set_ylabel("Score (Macro-Averaged)", fontsize=9.5, color="#1A202C")
    ax.set_title(
        f"Model Evaluation Comparison on Test Split (N={n_test}, 95% CI Error Bars)",
        fontsize=10.5,
        fontweight="bold",
        color="#1B365D",
        pad=26,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.xaxis.grid(False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig("accuracy_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if _has_per_class_metrics(baseline_metrics):
        _generate_per_class_chart(baseline_metrics)


def _generate_per_class_chart(baseline_metrics: Dict[str, Any]) -> None:
    """Generate Fig. 2 showing per-class Precision, Recall, and F1 for baselines."""
    lr0 = baseline_metrics["lr_report"]["0"]
    lr1 = baseline_metrics["lr_report"]["1"]
    nb0 = baseline_metrics["nb_report"]["0"]
    nb1 = baseline_metrics["nb_report"]["1"]

    categories = [
        f"TF-IDF+LR\nNeg (n={int(lr0['support'])})",
        f"TF-IDF+LR\nPos (n={int(lr1['support'])})",
        f"Multinomial NB\nNeg (n={int(nb0['support'])})",
        f"Multinomial NB\nPos (n={int(nb1['support'])})",
    ]
    cls_dicts = [lr0, lr1, nb0, nb1]
    metric_keys = [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1-Score")]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(categories))
    width = 0.24
    palette = sns.color_palette("colorblind", len(metric_keys))

    for i, (m_key, m_label) in enumerate(metric_keys):
        vals = [float(d[m_key]) for d in cls_dicts]
        bars = ax.bar(
            x + (i - 1.0) * width,
            vals,
            width,
            label=m_label,
            color=palette[i],
            edgecolor="#2D3748",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#1A202C",
            )

    min_cls_score = min(
        float(d[m_key]) for d in cls_dicts for m_key, _ in metric_keys
    )
    ax.set_ylim(0.50 if min_cls_score >= 0.60 else 0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.8, fontweight="bold", color="#1A202C")
    ax.set_ylabel("Class Score", fontsize=9.5, color="#1A202C")
    ax.set_title(
        "Heuristic Baseline Class-Level Precision, Recall, and F1 Breakdown",
        fontsize=10.5,
        fontweight="bold",
        color="#1B365D",
        pad=24,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.xaxis.grid(False)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig("class_metrics_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_page_chrome(canvas: Canvas, doc: BaseDocTemplate) -> None:
    """Render running header and footer with page numbers."""
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, height - 0.48 * inch, width - doc.rightMargin, height - 0.48 * inch)
    canvas.line(doc.leftMargin, 0.48 * inch, width - doc.rightMargin, 0.48 * inch)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(doc.leftMargin, height - 0.38 * inch, "E-Commerce Sentiment Analysis & Triage — Model QA Report")
        canvas.drawRightString(width - doc.rightMargin, height - 0.38 * inch, "Dataset: fancyzhx/amazon_polarity")

    canvas.drawString(doc.leftMargin, 0.32 * inch, "Metrics sourced dynamically from JSON evaluation artifacts")
    canvas.drawRightString(width - doc.rightMargin, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _format_pvalue(p_val: float) -> str:
    """Format p-value clearly without claiming p=0.0000."""
    if p_val < 0.0001:
        return "p < 0.0001"
    return f"p = {p_val:.4f}"


def build_pdf_report() -> None:
    """Validate JSON artifacts, render charts, and build report.pdf."""
    print("Generating Data Science & Model QA Evaluation Report...")

    try:
        baseline_metrics = load_and_validate_metrics("baseline_metrics.json")
        lda_topics_raw = load_and_validate_metrics("lda_topics.json")
        lda_topics: Dict[int, str] = {}
        for idx, (k, v) in enumerate(lda_topics_raw.items()):
            try:
                topic_key = int(k)
            except (TypeError, ValueError):
                topic_key = idx
            lda_topics[topic_key] = str(v)
        transformer_metrics = load_and_validate_metrics("transformer_metrics.json")

        raw_n_test = baseline_metrics.get(
            "n_test_samples",
            baseline_metrics.get("lr_report", {}).get("macro avg", {}).get("support", 400),
        )
        if not _is_valid_number(raw_n_test) or int(raw_n_test) <= 0:
            raise ValueError(
                f"Critical Data Error: n_test_samples ({raw_n_test}) must be a positive finite integer."
            )
        n_test = int(raw_n_test)

        neg_support = float(baseline_metrics.get("lr_report", {}).get("0", {}).get("support", 0))
        pos_support = float(baseline_metrics.get("lr_report", {}).get("1", {}).get("support", 0))
        total_support = neg_support + pos_support
        if total_support > 0:
            neg_pct = (neg_support / total_support) * 100.0
            pos_pct = (pos_support / total_support) * 100.0
        else:
            neg_pct = 50.0
            pos_pct = 50.0

        lr_acc = baseline_metrics["lr_accuracy"]
        lr_p = baseline_metrics["lr_report"]["macro avg"]["precision"]
        lr_r = baseline_metrics["lr_report"]["macro avg"]["recall"]
        lr_f1 = baseline_metrics["lr_report"]["macro avg"]["f1-score"]

        nb_acc = baseline_metrics["nb_accuracy"]
        nb_p = baseline_metrics["nb_report"]["macro avg"]["precision"]
        nb_r = baseline_metrics["nb_report"]["macro avg"]["recall"]
        nb_f1 = baseline_metrics["nb_report"]["macro avg"]["f1-score"]

        t_acc = transformer_metrics["eval_accuracy"]
        t_p = transformer_metrics["eval_precision"]
        t_r = transformer_metrics["eval_recall"]
        t_f1 = transformer_metrics["eval_f1"]

        metrics_to_validate = [lr_acc, lr_p, lr_r, lr_f1, nb_acc, nb_p, nb_r, nb_f1, t_acc, t_p, t_r, t_f1]
        for val in metrics_to_validate:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"Critical Data Error: Metric {val} is not a numeric type.")
            if not math.isfinite(float(val)) or not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"Critical Data Error: Metric {val} is outside valid 0.0 - 1.0 bounds.")

        lr_acc_f, lr_p_f, lr_r_f, lr_f1_f = float(lr_acc), float(lr_p), float(lr_r), float(lr_f1)
        nb_acc_f, nb_p_f, nb_r_f, nb_f1_f = float(nb_acc), float(nb_p), float(nb_r), float(nb_f1)
        t_acc_f, t_p_f, t_r_f, t_f1_f = float(t_acc), float(t_p), float(t_r), float(t_f1)
    except KeyError as e:
        raise KeyError(f"Missing expected key in JSON metrics: {str(e)}")

    generate_charts(baseline_metrics, transformer_metrics, n_test)
    has_per_class = _has_per_class_metrics(baseline_metrics)

    # Derived statistical comparisons
    z_lr, p_lr = calc_two_proportion_ztest(t_acc_f, lr_acc_f, n_test, n_test)
    z_nb, p_nb = calc_two_proportion_ztest(t_acc_f, nb_acc_f, n_test, n_test)
    f1_lift_lr_pp = (t_f1_f - lr_f1_f) * 100.0
    acc_lift_lr_pp = (t_acc_f - lr_acc_f) * 100.0
    err_red_lr = ((1.0 - lr_acc_f) - (1.0 - t_acc_f)) / (1.0 - lr_acc_f) * 100.0 if lr_acc_f < 1.0 else 0.0
    err_red_nb = ((1.0 - nb_acc_f) - (1.0 - t_acc_f)) / (1.0 - nb_acc_f) * 100.0 if nb_acc_f < 1.0 else 0.0

    eval_loss = transformer_metrics.get("eval_loss")
    eval_runtime = transformer_metrics.get("eval_runtime")
    eval_sps = transformer_metrics.get("eval_samples_per_second")
    eval_epoch = transformer_metrics.get("epoch")

    # Document setup (single-column letter)
    margin = 0.60 * inch
    doc = BaseDocTemplate(
        "report.pdf",
        pagesize=letter,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=0.62 * inch,
        bottomMargin=0.60 * inch,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="Report", frames=[frame], onPage=_draw_page_chrome)])
    usable_width = doc.width

    # Typography styles
    s_title = ParagraphStyle("RepTitle", fontName="Helvetica-Bold", fontSize=17.5, leading=21, textColor=NAVY, spaceAfter=3)
    s_subtitle = ParagraphStyle("RepSub", fontName="Helvetica", fontSize=10, leading=13, textColor=SLATE, spaceAfter=6)
    s_meta = ParagraphStyle("RepMeta", fontName="Helvetica", fontSize=8.3, leading=11, textColor=MUTED, spaceAfter=10)
    s_h1 = ParagraphStyle("RepH1", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=4)
    s_h2 = ParagraphStyle("RepH2", fontName="Helvetica-Bold", fontSize=9.8, leading=12.5, textColor=ACCENT_BLUE, spaceBefore=6, spaceAfter=3)
    s_body = ParagraphStyle("RepBody", fontName="Helvetica", fontSize=9.3, leading=13, textColor=DARK_TEXT, alignment=TA_JUSTIFY, spaceAfter=5)
    s_bullet = ParagraphStyle("RepBullet", fontName="Helvetica", fontSize=9.0, leading=12.2, textColor=DARK_TEXT, leftIndent=12, firstLineIndent=-8, spaceAfter=2.0)
    s_caption = ParagraphStyle("RepCaption", fontName="Helvetica-Oblique", fontSize=8.2, leading=10.5, textColor=SLATE, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)
    s_cell = ParagraphStyle("RepCell", fontName="Helvetica", fontSize=8.3, leading=10.5, textColor=DARK_TEXT)
    s_cell_hdr = ParagraphStyle("RepCellHdr", fontName="Helvetica-Bold", fontSize=8.3, leading=10.5, textColor=colors.white)
    s_kpi_lbl = ParagraphStyle("KpiLbl", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=SLATE, alignment=TA_CENTER)
    s_kpi_val = ParagraphStyle("KpiVal", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY, alignment=TA_CENTER)
    s_kpi_sub = ParagraphStyle("KpiSub", fontName="Helvetica", fontSize=7.5, leading=10, textColor=MUTED, alignment=TA_CENTER)
    s_ref = ParagraphStyle("RepRef", fontName="Helvetica", fontSize=8.2, leading=11, textColor=SLATE, spaceAfter=2.5)

    story: List[Any] = []

    # Header block
    story.append(Paragraph("E-Commerce Sentiment Classification &amp; Complaint Triage", s_title))
    story.append(Paragraph("Comparative Benchmarking of Heuristic Baselines vs. Fine-Tuned DistilBERT with LDA Topic Discovery", s_subtitle))
    story.append(
        Paragraph(
            f"<b>Author:</b> William (wiliamzwili135@gmail.com) &nbsp;|&nbsp; "
            f"<b>Dataset:</b> fancyzhx/amazon_polarity &nbsp;|&nbsp; "
            f"<b>Test Split:</b> N={n_test} ({pos_pct:.2f}% Pos / {neg_pct:.2f}% Neg) &nbsp;|&nbsp; "
            f"<b>Seed:</b> 42",
            s_meta,
        )
    )

    # Executive Summary
    story.append(Paragraph("Executive Summary", s_h1))
    story.append(
        Paragraph(
            f"We benchmarked two TF-IDF heuristic baselines (Logistic Regression and Multinomial Naive Bayes) against a "
            f"fine-tuned <font face='Courier'>distilbert-base-uncased</font> classifier on N={n_test} held-out Amazon product "
            f"reviews, paired with k={len(lda_topics)}-topic Latent Dirichlet Allocation (LDA) on negative training reviews. "
            f"DistilBERT achieved <b>{t_f1_f:.3f}</b> macro F1 and <b>{t_acc_f:.3f}</b> accuracy, outperforming TF-IDF+LR "
            f"by <b>{f1_lift_lr_pp:+.2f} pp</b> macro F1 and reducing classification error rate by <b>{err_red_lr:.1f}%</b> "
            f"(two-proportion z={z_lr:.2f}, {_format_pvalue(p_lr)}). Unsupervised LDA surfaced {len(lda_topics)} recurring "
            f"complaint clusters spanning gaming/software media, music CDs, books/film, general product value, and hardware/connector failures.",
            s_body,
        )
    )

    # 4-Card KPI Strip
    kpi_4_lbl = "EVAL THROUGHPUT" if _is_valid_number(eval_sps) else "EVALUATED TEST N"
    kpi_4_val = f"{float(eval_sps):.1f} rev/s" if _is_valid_number(eval_sps) else f"N={n_test}"
    kpi_4_sub = (
        f"loss={float(eval_loss):.3f}, {float(eval_runtime):.1f}s"
        if (_is_valid_number(eval_loss) and _is_valid_number(eval_runtime))
        else f"{pos_pct:.1f}% pos / {neg_pct:.1f}% neg"
    )
    kpi_data = [
        [
            Paragraph("DISTILBERT MACRO F1", s_kpi_lbl),
            Paragraph("LIFT VS TF-IDF+LR", s_kpi_lbl),
            Paragraph("ERROR REDUCTION", s_kpi_lbl),
            Paragraph(kpi_4_lbl, s_kpi_lbl),
        ],
        [
            Paragraph(f"{t_f1_f:.3f}", s_kpi_val),
            Paragraph(f"{f1_lift_lr_pp:+.2f} pp", s_kpi_val),
            Paragraph(f"{err_red_lr:.1f}%", s_kpi_val),
            Paragraph(kpi_4_val, s_kpi_val),
        ],
        [
            Paragraph(f"95% CI: &plusmn;{approx_composite_ci(t_f1_f, n_test):.3f}", s_kpi_sub),
            Paragraph(f"z={z_lr:.2f}, {_format_pvalue(p_lr)}", s_kpi_sub),
            Paragraph(f"{1.0 - lr_acc_f:.1%} &rarr; {1.0 - t_acc_f:.1%} err", s_kpi_sub),
            Paragraph(kpi_4_sub, s_kpi_sub),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[usable_width / 4.0] * 4)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(Spacer(1, 4))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # Section 1: Experimental Setup
    story.append(Paragraph("1. Experimental Setup &amp; Pipeline Architecture", s_h1))
    story.append(
        Paragraph(
            f"All models were trained and evaluated deterministically (seed=42) on binary review sentiment labels "
            f"(0=Negative, 1=Positive) extracted from <font face='Courier'>fancyzhx/amazon_polarity</font>. Text preprocessing "
            f"(<font face='Courier'>clean_text</font>) strips HTML tags, removes non-alphabetic characters and digits, lowercases, "
            f"and normalizes whitespace. Under default <font face='Courier'>DataConfig</font>, 2,000 training and 400 test reviews are "
            f"sampled; the evaluated test split here has N={n_test} samples ({int(pos_support)} positive, {int(neg_support)} negative).",
            s_body,
        )
    )
    setup_rows = [
        [Paragraph("Component", s_cell_hdr), Paragraph("Configuration / Hyperparameters", s_cell_hdr), Paragraph("Role in Pipeline", s_cell_hdr)],
        [Paragraph("TF-IDF + Logistic Regression", s_cell), Paragraph("TfidfVectorizer(max_features=5000, stop_words='english'); LR(max_iter=1000, seed=42)", s_cell), Paragraph("Linear bag-of-words baseline", s_cell)],
        [Paragraph("TF-IDF + Multinomial NB", s_cell), Paragraph("Shared 5,000-term TF-IDF matrix; MultinomialNB() default smoothing", s_cell), Paragraph("Probabilistic generative baseline", s_cell)],
        [Paragraph("DistilBERT Fine-Tune", s_cell), Paragraph(f"distilbert-base-uncased, max_length=128, lr=2e-5, batch=8, epochs={int(eval_epoch) if _is_valid_number(eval_epoch) else 1}, weight_decay=0.01", s_cell), Paragraph("Contextual attention classifier", s_cell)],
        [Paragraph("Negative-Review LDA", s_cell), Paragraph(f"CountVectorizer(max_df=0.95, min_df=2, English stop words); LDA(k={len(lda_topics)}, online, max_iter=5)", s_cell), Paragraph("Complaint topic discovery on negative train set", s_cell)],
    ]
    setup_table = Table(setup_rows, colWidths=[1.55 * inch, 3.45 * inch, usable_width - 5.0 * inch])
    setup_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
                ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(KeepTogether([setup_table, Paragraph("Table I. Pipeline configuration and model specification summary.", s_caption)]))

    # Section 2: Model Evaluation & Statistical Analysis
    story.append(Paragraph("2. Model Evaluation &amp; Statistical Comparison", s_h1))
    story.append(
        Paragraph(
            "Table II reports macro-averaged Precision, Recall, F1-Score, and overall Accuracy across all three architectures. "
            "Accuracy CIs use standard binomial intervals (1.96 * sqrt(p(1-p)/N)); composite Precision, Recall, and F1 CIs apply a "
            "conservative Variance Inflation Factor (VIF=1.5) because per-sample test predictions for exact bootstrap CIs are not persisted.",
            s_body,
        )
    )

    t_extra = (
        f"loss={float(eval_loss):.3f}, {float(eval_runtime):.2f}s"
        if (_is_valid_number(eval_loss) and _is_valid_number(eval_runtime))
        else "N/A"
    )
    perf_rows = [
        [
            Paragraph("Model", s_cell_hdr),
            Paragraph("Precision (Macro)", s_cell_hdr),
            Paragraph("Recall (Macro)", s_cell_hdr),
            Paragraph("F1-Score (Macro)", s_cell_hdr),
            Paragraph("Accuracy", s_cell_hdr),
            Paragraph("Notes", s_cell_hdr),
        ],
        [
            Paragraph("TF-IDF + LR", s_cell),
            Paragraph(f"{lr_p_f:.3f} &plusmn; {approx_composite_ci(lr_p_f, n_test):.3f}", s_cell),
            Paragraph(f"{lr_r_f:.3f} &plusmn; {approx_composite_ci(lr_r_f, n_test):.3f}", s_cell),
            Paragraph(f"{lr_f1_f:.3f} &plusmn; {approx_composite_ci(lr_f1_f, n_test):.3f}", s_cell),
            Paragraph(f"{lr_acc_f:.3f} &plusmn; {calc_accuracy_ci(lr_acc_f, n_test):.3f}", s_cell),
            Paragraph("Linear baseline", s_cell),
        ],
        [
            Paragraph("Multinomial NB", s_cell),
            Paragraph(f"{nb_p_f:.3f} &plusmn; {approx_composite_ci(nb_p_f, n_test):.3f}", s_cell),
            Paragraph(f"{nb_r_f:.3f} &plusmn; {approx_composite_ci(nb_r_f, n_test):.3f}", s_cell),
            Paragraph(f"{nb_f1_f:.3f} &plusmn; {approx_composite_ci(nb_f1_f, n_test):.3f}", s_cell),
            Paragraph(f"{nb_acc_f:.3f} &plusmn; {calc_accuracy_ci(nb_acc_f, n_test):.3f}", s_cell),
            Paragraph("Naive Bayes baseline", s_cell),
        ],
        [
            Paragraph("<b>DistilBERT</b>", s_cell),
            Paragraph(f"<b>{t_p_f:.3f} &plusmn; {approx_composite_ci(t_p_f, n_test):.3f}</b>", s_cell),
            Paragraph(f"<b>{t_r_f:.3f} &plusmn; {approx_composite_ci(t_r_f, n_test):.3f}</b>", s_cell),
            Paragraph(f"<b>{t_f1_f:.3f} &plusmn; {approx_composite_ci(t_f1_f, n_test):.3f}</b>", s_cell),
            Paragraph(f"<b>{t_acc_f:.3f} &plusmn; {calc_accuracy_ci(t_acc_f, n_test):.3f}</b>", s_cell),
            Paragraph(t_extra, s_cell),
        ],
    ]
    perf_table = Table(
        perf_rows,
        colWidths=[1.15 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.10 * inch, usable_width - 5.70 * inch],
    )
    perf_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, 2), [colors.white, ALT_ROW]),
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#EBF4FF")),
                ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN", (1, 1), (4, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(KeepTogether([perf_table, Paragraph(f"Table II. Macro-averaged performance on N={n_test} test reviews with 95% CIs.", s_caption)]))

    fig1_img = Image("accuracy_chart.png", width=6.4 * inch, height=3.1 * inch)
    story.append(KeepTogether([fig1_img, Paragraph("Fig. 1. Macro-averaged Accuracy, Precision, Recall, and F1-Score with 95% confidence intervals.", s_caption)]))

    story.append(Paragraph("2.1 Statistical Significance &amp; Error-Rate Reduction", s_h2))
    story.append(
        Paragraph(
            f"Under an independent two-proportion z-test (used as an approximation without paired prediction vectors for McNemar's test), "
            f"DistilBERT's accuracy gain over TF-IDF+LR ({acc_lift_lr_pp:+.2f} pp, {lr_acc_f:.3f} &rarr; {t_acc_f:.3f}) yields "
            f"<b>z = {z_lr:.2f} ({_format_pvalue(p_lr)})</b>, corresponding to a <b>{err_red_lr:.1f}%</b> relative error reduction. "
            f"Compared against Multinomial NB ({nb_acc_f:.3f} &rarr; {t_acc_f:.3f}), DistilBERT achieves a "
            f"<b>{(t_acc_f - nb_acc_f) * 100.0:+.2f} pp</b> gain (<b>z = {z_nb:.2f}, {_format_pvalue(p_nb)}</b>; "
            f"<b>{err_red_nb:.1f}%</b> relative error reduction). Attentional representations resolve negation and multi-word sentiment "
            f"reversals that bag-of-words n-grams lose.",
            s_body,
        )
    )

    if has_per_class:
        lr0 = baseline_metrics["lr_report"]["0"]
        lr1 = baseline_metrics["lr_report"]["1"]
        nb0 = baseline_metrics["nb_report"]["0"]
        nb1 = baseline_metrics["nb_report"]["1"]
        story.append(Paragraph("2.2 Heuristic Class-Level Breakdown", s_h2))
        story.append(
            Paragraph(
                f"Per-class inspection of the heuristic baselines (Table III, Fig. 2) shows balanced behavior for TF-IDF+LR "
                f"(Negative recall {float(lr0['recall']):.3f} vs. Positive recall {float(lr1['recall']):.3f}), whereas Multinomial NB "
                f"exhibits slightly higher Negative recall ({float(nb0['recall']):.3f}) alongside lower Negative precision "
                f"({float(nb0['precision']):.3f}), indicating false-negative drift on nuanced positive reviews. "
                f"Note that <font face='Courier'>transformer_metrics.json</font> logs macro-averaged summary metrics only, so "
                f"per-class DistilBERT confusion breakdowns require persisting test predictions in a future run.",
                s_body,
            )
        )
        cls_rows = [
            [Paragraph("Model", s_cell_hdr), Paragraph("Class", s_cell_hdr), Paragraph("Precision", s_cell_hdr), Paragraph("Recall", s_cell_hdr), Paragraph("F1-Score", s_cell_hdr), Paragraph("Support", s_cell_hdr)],
            [Paragraph("TF-IDF + LR", s_cell), Paragraph("0 (Negative)", s_cell), Paragraph(f"{float(lr0['precision']):.3f}", s_cell), Paragraph(f"{float(lr0['recall']):.3f}", s_cell), Paragraph(f"{float(lr0['f1-score']):.3f}", s_cell), Paragraph(f"{int(lr0['support'])}", s_cell)],
            [Paragraph("TF-IDF + LR", s_cell), Paragraph("1 (Positive)", s_cell), Paragraph(f"{float(lr1['precision']):.3f}", s_cell), Paragraph(f"{float(lr1['recall']):.3f}", s_cell), Paragraph(f"{float(lr1['f1-score']):.3f}", s_cell), Paragraph(f"{int(lr1['support'])}", s_cell)],
            [Paragraph("Multinomial NB", s_cell), Paragraph("0 (Negative)", s_cell), Paragraph(f"{float(nb0['precision']):.3f}", s_cell), Paragraph(f"{float(nb0['recall']):.3f}", s_cell), Paragraph(f"{float(nb0['f1-score']):.3f}", s_cell), Paragraph(f"{int(nb0['support'])}", s_cell)],
            [Paragraph("Multinomial NB", s_cell), Paragraph("1 (Positive)", s_cell), Paragraph(f"{float(nb1['precision']):.3f}", s_cell), Paragraph(f"{float(nb1['recall']):.3f}", s_cell), Paragraph(f"{float(nb1['f1-score']):.3f}", s_cell), Paragraph(f"{int(nb1['support'])}", s_cell)],
        ]
        cls_table = Table(cls_rows, colWidths=[1.35 * inch, 1.15 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch, usable_width - 5.65 * inch])
        cls_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
                    ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(KeepTogether([cls_table, Paragraph("Table III. Per-class precision, recall, F1, and support for heuristic baselines.", s_caption)]))
        fig2_img = Image("class_metrics_chart.png", width=5.6 * inch, height=2.41 * inch)
        story.append(KeepTogether([fig2_img, Paragraph("Fig. 2. Per-class metric comparison across Negative and Positive review splits.", s_caption)]))

    # Section 3: LDA Topic Discovery
    story.append(Paragraph("3. Unsupervised Complaint Topic Discovery (LDA)", s_h1))
    story.append(
        Paragraph(
            f"LDA (k={len(lda_topics)} components, online variational Bayes) was fit on negative reviews from the training split "
            f"(<font face='Courier'>label == 0</font>) to surface recurring complaint themes without manual tagging.",
            s_body,
        )
    )
    topic_rows = [[Paragraph("Topic", s_cell_hdr), Paragraph("Top 10 Weighted Terms", s_cell_hdr)]]
    for topic_id in sorted(lda_topics.keys()):
        terms_escaped = escape(str(lda_topics[topic_id]), quote=False)
        topic_rows.append([Paragraph(f"Topic {topic_id + 1}", s_cell), Paragraph(terms_escaped, s_cell)])

    topic_table = Table(topic_rows, colWidths=[0.95 * inch, usable_width - 0.95 * inch])
    topic_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
                ("BOX", (0, 0), (-1, -1), 0.75, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    table_num_lda = "Table IV" if has_per_class else "Table III"
    story.append(KeepTogether([topic_table, Paragraph(f"{table_num_lda}. Top terms extracted via LDA on negative training reviews (k={len(lda_topics)}).", s_caption)]))

    all_topic_tokens = " ".join(lda_topics.values()).split()
    contraction_artifacts = [tok for tok in ("dont", "isnt", "does", "didnt", "cant") if tok in all_topic_tokens]
    artifact_note = (
        f" Additionally, apostrophe stripping in <font face='Courier'>clean_text</font> produces contraction fragments "
        f"({', '.join(contraction_artifacts)}) that bypass standard English stop-word filtering; expanding contractions before "
        f"punctuation removal would tighten topic coherence."
        if contraction_artifacts
        else ""
    )
    story.append(
        Paragraph(
            "The extracted clusters separate product categories and failure modes—such as video games/discs, music albums/CDs, "
            "books/movies, general product value, and hardware/connector defects (e.g., switch, disk, connector, laptop)."
            + artifact_note,
            s_body,
        )
    )

    # Section 4: Model QA, Limitations & Recommendations
    story.append(Paragraph("4. Model QA: Limitations, Risks &amp; Recommendations", s_h1))
    qa_items = [
        f"<b>Subsample Variance (N={n_test}):</b> Test metrics carry ~&plusmn;3–5% 95% confidence intervals due to the 2,000-train / {n_test}-test subsample. Scale evaluation to &ge;10,000 test reviews before production SLAs.",
        "<b>Missing Per-Sample Predictions &amp; Calibration:</b> Because raw test probabilities are not logged, ROC-AUC, Brier score / reliability calibration, and exact paired McNemar tests cannot be computed from current JSON artifacts.",
        "<b>Single-Epoch Fine-Tuning:</b> DistilBERT was trained for 1 epoch without a dedicated validation split or hyperparameter sweep; early stopping on a 3-way train/val/test split is recommended.",
        f"<b>Static Topic Count (k={len(lda_topics)}):</b> Topic count was fixed rather than selected via C_v coherence or perplexity curves, and <font face='Courier'>clean_text</font> strips contractions and negations alongside stop words.",
        f"<b>Latency vs. Accuracy Trade-Off:</b> DistilBERT inference ({kpi_4_val}) requires GPU/accelerator batching for high-throughput triage, whereas TF-IDF+LR runs in sub-millisecond CPU time and can serve as a low-latency confidence-gated fallback.",
    ]
    for item in qa_items:
        story.append(Paragraph(f"&bull; &nbsp; {item}", s_bullet))

    # Section 5: References
    story.append(Paragraph("5. References", s_h1))
    refs = [
        "[1] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,\" <i>arXiv:1910.01108</i>, 2019.",
        "[2] A. Vaswani et al., \"Attention is all you need,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2017.",
        "[3] D. M. Blei, A. Y. Ng, and M. I. Jordan, \"Latent Dirichlet Allocation,\" <i>Journal of Machine Learning Research</i>, vol. 3, pp. 993–1022, 2003.",
        "[4] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, s_ref))

    doc.build(story)
    print("Report generated successfully as 'report.pdf'")


if __name__ == "__main__":
    build_pdf_report()
