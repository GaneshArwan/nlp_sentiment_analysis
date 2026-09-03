import streamlit as st
import json
import pandas as pd
import numpy as np

# UI Configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Removed @st.cache_data to prevent stale data and permanent error lockouts
def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        # Don't show st.error here to avoid cluttering if files just aren't generated yet
        return None

def calc_ci(p, n):
    n = max(int(n), 1)  # Prevent ZeroDivisionError
    return 1.96 * np.sqrt((p * (1 - p)) / n)
    
def calc_composite_ci(p, n):
    n = max(int(n), 1)  # Prevent ZeroDivisionError
    return 1.96 * np.sqrt(1.5 * (p * (1 - p)) / n)

def main():
    st.title("📊 NLP Sentiment Analysis Dashboard")
    st.markdown("Comparative Evaluation of Transformer vs. Heuristic Models in E-Commerce")
    
    baseline_metrics = load_data('baseline_metrics.json') or {}
    lda_topics = load_data('lda_topics.json') or {}
    transformer_metrics = load_data('transformer_metrics.json') or {}
    
    if not (baseline_metrics and lda_topics and transformer_metrics):
        st.warning("Awaiting metric files generation. Please run the training pipeline first.")
        return
        
    n_test = baseline_metrics.get('n_test_samples', 2000)
    
    tab1, tab2, tab3 = st.tabs(["🚀 Model Performance", "🔍 Topic Extraction", "ℹ️ Methodology"])
    
    with tab1:
        st.header("Performance Overview")
        
        t_f1 = transformer_metrics.get('eval_f1', 0.0)
        t_acc = transformer_metrics.get('eval_accuracy', 0.0)
        
        lr_report = baseline_metrics.get('lr_report', {}).get('macro avg', {})
        lr_f1 = lr_report.get('f1-score', 0.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="DistilBERT F1-Score", value=f"{t_f1:.3f}", delta=f"+{t_f1 - lr_f1:.3f} vs TF-IDF")
        with col2:
            st.metric(label="DistilBERT Accuracy", value=f"{t_acc:.3f}")
        with col3:
            st.metric(label="Evaluation Samples (N)", value=n_test)
            
        st.markdown("---")
        st.subheader("Detailed Macro-Averaged Metrics")
        
        lr_acc = baseline_metrics.get('lr_accuracy', 0.0)
        lr_p = lr_report.get('precision', 0.0)
        lr_r = lr_report.get('recall', 0.0)
        
        nb_report = baseline_metrics.get('nb_report', {}).get('macro avg', {})
        nb_acc = baseline_metrics.get('nb_accuracy', 0.0)
        nb_p = nb_report.get('precision', 0.0)
        nb_r = nb_report.get('recall', 0.0)
        nb_f1 = nb_report.get('f1-score', 0.0)
        
        t_p = transformer_metrics.get('eval_precision', 0.0)
        t_r = transformer_metrics.get('eval_recall', 0.0)
        
        df_metrics = pd.DataFrame([
            {"Model": "TF-IDF + LR", "Precision": f"{lr_p:.3f} ± {calc_composite_ci(lr_p, n_test):.3f}", "Recall": f"{lr_r:.3f} ± {calc_composite_ci(lr_r, n_test):.3f}", "F1": f"{lr_f1:.3f} ± {calc_composite_ci(lr_f1, n_test):.3f}", "Accuracy": f"{lr_acc:.3f} ± {calc_ci(lr_acc, n_test):.3f}"},
            {"Model": "Multinomial NB", "Precision": f"{nb_p:.3f} ± {calc_composite_ci(nb_p, n_test):.3f}", "Recall": f"{nb_r:.3f} ± {calc_composite_ci(nb_r, n_test):.3f}", "F1": f"{nb_f1:.3f} ± {calc_composite_ci(nb_f1, n_test):.3f}", "Accuracy": f"{nb_acc:.3f} ± {calc_ci(nb_acc, n_test):.3f}"},
            {"Model": "DistilBERT", "Precision": f"{t_p:.3f} ± {calc_composite_ci(t_p, n_test):.3f}", "Recall": f"{t_r:.3f} ± {calc_composite_ci(t_r, n_test):.3f}", "F1": f"{t_f1:.3f} ± {calc_composite_ci(t_f1, n_test):.3f}", "Accuracy": f"{t_acc:.3f} ± {calc_ci(t_acc, n_test):.3f}"}
        ])
        
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
    with tab2:
        st.header("Negative Sentiment Thematic Analysis")
        st.info("Latent Dirichlet Allocation (LDA) applied to negatively classified feedback to extract actionable domains.")
        
        for k, v in lda_topics.items():
            st.markdown(f"**Topic {int(k)+1}**: `{v}`")
            
    with tab3:
        st.header("Pipeline Methodology")
        st.markdown('''
        * **Heuristics**: TF-IDF vectorization (max_features=5000) evaluated via Logistic Regression and Multinomial Naive Bayes.
        * **Transformer**: Fine-tuned `distilbert-base-uncased` utilizing HuggingFace Trainer API.
        * **Metrics**: Macro-averaged Precision, Recall, and F1-score to mitigate class imbalance biases. 95% Confidence Intervals calculated using variance inflation for composite metrics.
        ''')

if __name__ == "__main__":
    main()
