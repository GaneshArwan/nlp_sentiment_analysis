import json
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# UI Configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _is_valid_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))

# Removed @st.cache_data to prevent stale data and permanent error lockouts
def load_data(filepath: str) -> Optional[Dict[str, Any]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        # Don't show st.error here to avoid cluttering if files just aren't generated yet
        return None

def calc_ci(p: float, n: int) -> float:
    if not _is_valid_number(p):
        return 0.0
    p_clamped = min(max(float(p), 0.0), 1.0)
    n_safe = max(int(n), 1)  # Prevent ZeroDivisionError
    return float(1.96 * np.sqrt((p_clamped * (1.0 - p_clamped)) / n_safe))

def calc_composite_ci(p: float, n: int) -> float:
    if not _is_valid_number(p):
        return 0.0
    p_clamped = min(max(float(p), 0.0), 1.0)
    n_safe = max(int(n), 1)  # Prevent ZeroDivisionError
    return float(1.96 * np.sqrt(1.5 * (p_clamped * (1.0 - p_clamped)) / n_safe))

def main() -> None:
    st.title("📊 NLP Sentiment Analysis Dashboard")
    st.markdown("Comparative Evaluation of Transformer vs. Heuristic Models in E-Commerce")
    
    baseline_metrics = load_data('baseline_metrics.json')
    lda_topics = load_data('lda_topics.json')
    transformer_metrics = load_data('transformer_metrics.json')
    
    if not (baseline_metrics and lda_topics and transformer_metrics):
        st.warning("Awaiting metric files generation. Please run the training pipeline first.")
        return
        
    raw_n_test = baseline_metrics.get(
        'n_test_samples',
        baseline_metrics.get('lr_report', {}).get('macro avg', {}).get('support', 400),
    )
    n_test = max(int(raw_n_test), 1) if _is_valid_number(raw_n_test) else 400
    
    tab1, tab2, tab3 = st.tabs(["🚀 Model Performance", "🔍 Topic Extraction", "ℹ️ Methodology"])
    
    with tab1:
        st.header("Performance Overview")
        
        t_f1 = float(transformer_metrics.get('eval_f1', 0.0)) if _is_valid_number(transformer_metrics.get('eval_f1')) else 0.0
        t_acc = float(transformer_metrics.get('eval_accuracy', 0.0)) if _is_valid_number(transformer_metrics.get('eval_accuracy')) else 0.0
        
        lr_report = baseline_metrics.get('lr_report', {}).get('macro avg', {})
        lr_f1 = float(lr_report.get('f1-score', 0.0)) if _is_valid_number(lr_report.get('f1-score')) else 0.0
        lr_acc = float(baseline_metrics.get('lr_accuracy', 0.0)) if _is_valid_number(baseline_metrics.get('lr_accuracy')) else 0.0
        err_red = ((1.0 - lr_acc) - (1.0 - t_acc)) / (1.0 - lr_acc) * 100.0 if lr_acc < 1.0 else 0.0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="DistilBERT F1-Score", value=f"{t_f1:.3f}", delta=f"{t_f1 - lr_f1:+.3f} vs TF-IDF")
        with col2:
            st.metric(label="DistilBERT Accuracy", value=f"{t_acc:.3f}")
        with col3:
            st.metric(label="Relative Error Reduction", value=f"{err_red:.1f}%")
        with col4:
            st.metric(label="Evaluation Samples (N)", value=n_test)
            
        st.markdown("---")
        st.subheader("Detailed Macro-Averaged Metrics")
        
        lr_p = float(lr_report.get('precision', 0.0)) if _is_valid_number(lr_report.get('precision')) else 0.0
        lr_r = float(lr_report.get('recall', 0.0)) if _is_valid_number(lr_report.get('recall')) else 0.0
        
        nb_report = baseline_metrics.get('nb_report', {}).get('macro avg', {})
        nb_acc = float(baseline_metrics.get('nb_accuracy', 0.0)) if _is_valid_number(baseline_metrics.get('nb_accuracy')) else 0.0
        nb_p = float(nb_report.get('precision', 0.0)) if _is_valid_number(nb_report.get('precision')) else 0.0
        nb_r = float(nb_report.get('recall', 0.0)) if _is_valid_number(nb_report.get('recall')) else 0.0
        nb_f1 = float(nb_report.get('f1-score', 0.0)) if _is_valid_number(nb_report.get('f1-score')) else 0.0
        
        t_p = float(transformer_metrics.get('eval_precision', 0.0)) if _is_valid_number(transformer_metrics.get('eval_precision')) else 0.0
        t_r = float(transformer_metrics.get('eval_recall', 0.0)) if _is_valid_number(transformer_metrics.get('eval_recall')) else 0.0
        
        df_metrics = pd.DataFrame([
            {"Model": "TF-IDF + LR", "Precision": f"{lr_p:.3f} ± {calc_composite_ci(lr_p, n_test):.3f}", "Recall": f"{lr_r:.3f} ± {calc_composite_ci(lr_r, n_test):.3f}", "F1": f"{lr_f1:.3f} ± {calc_composite_ci(lr_f1, n_test):.3f}", "Accuracy": f"{lr_acc:.3f} ± {calc_ci(lr_acc, n_test):.3f}"},
            {"Model": "Multinomial NB", "Precision": f"{nb_p:.3f} ± {calc_composite_ci(nb_p, n_test):.3f}", "Recall": f"{nb_r:.3f} ± {calc_composite_ci(nb_r, n_test):.3f}", "F1": f"{nb_f1:.3f} ± {calc_composite_ci(nb_f1, n_test):.3f}", "Accuracy": f"{nb_acc:.3f} ± {calc_ci(nb_acc, n_test):.3f}"},
            {"Model": "DistilBERT", "Precision": f"{t_p:.3f} ± {calc_composite_ci(t_p, n_test):.3f}", "Recall": f"{t_r:.3f} ± {calc_composite_ci(t_r, n_test):.3f}", "F1": f"{t_f1:.3f} ± {calc_composite_ci(t_f1, n_test):.3f}", "Accuracy": f"{t_acc:.3f} ± {calc_ci(t_acc, n_test):.3f}"}
        ])
        
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        if os.path.exists('accuracy_chart.png'):
            st.image('accuracy_chart.png', caption=f'Model Evaluation Comparison (N={n_test}, 95% CI)', use_container_width=True)

        lr_cls = baseline_metrics.get('lr_report', {})
        nb_cls = baseline_metrics.get('nb_report', {})
        if all(k in lr_cls and k in nb_cls for k in ('0', '1')):
            st.subheader("Heuristic Baseline Class-Level Breakdown")
            df_cls = pd.DataFrame([
                {"Model": "TF-IDF + LR", "Class": "0 (Negative)", "Precision": f"{float(lr_cls['0'].get('precision', 0.0)):.3f}", "Recall": f"{float(lr_cls['0'].get('recall', 0.0)):.3f}", "F1": f"{float(lr_cls['0'].get('f1-score', 0.0)):.3f}", "Support": int(lr_cls['0'].get('support', 0))},
                {"Model": "TF-IDF + LR", "Class": "1 (Positive)", "Precision": f"{float(lr_cls['1'].get('precision', 0.0)):.3f}", "Recall": f"{float(lr_cls['1'].get('recall', 0.0)):.3f}", "F1": f"{float(lr_cls['1'].get('f1-score', 0.0)):.3f}", "Support": int(lr_cls['1'].get('support', 0))},
                {"Model": "Multinomial NB", "Class": "0 (Negative)", "Precision": f"{float(nb_cls['0'].get('precision', 0.0)):.3f}", "Recall": f"{float(nb_cls['0'].get('recall', 0.0)):.3f}", "F1": f"{float(nb_cls['0'].get('f1-score', 0.0)):.3f}", "Support": int(nb_cls['0'].get('support', 0))},
                {"Model": "Multinomial NB", "Class": "1 (Positive)", "Precision": f"{float(nb_cls['1'].get('precision', 0.0)):.3f}", "Recall": f"{float(nb_cls['1'].get('recall', 0.0)):.3f}", "F1": f"{float(nb_cls['1'].get('f1-score', 0.0)):.3f}", "Support": int(nb_cls['1'].get('support', 0))},
            ])
            st.dataframe(df_cls, use_container_width=True, hide_index=True)
            if os.path.exists('class_metrics_chart.png'):
                st.image('class_metrics_chart.png', caption='Heuristic Baseline Class-Level Breakdown', use_container_width=True)
        
    with tab2:
        st.header("Negative Sentiment Thematic Analysis")
        st.info("Latent Dirichlet Allocation (LDA) applied to negatively classified feedback to extract actionable domains.")
        
        for idx, (k, v) in enumerate(lda_topics.items(), start=1):
            try:
                topic_num = int(k) + 1
            except (TypeError, ValueError):
                topic_num = idx
            st.markdown(f"**Topic {topic_num}**: `{v}`")
            
    with tab3:
        st.header("Pipeline Methodology")
        st.markdown('''
        * **Heuristics**: TF-IDF vectorization (max_features=5000) evaluated via Logistic Regression and Multinomial Naive Bayes.
        * **Transformer**: Fine-tuned `distilbert-base-uncased` utilizing HuggingFace Trainer API.
        * **Metrics**: Macro-averaged Precision, Recall, and F1-score to mitigate class imbalance biases. 95% Confidence Intervals calculated using variance inflation for composite metrics.
        ''')

if __name__ == "__main__":
    main()
