import streamlit as st
import json
import pandas as pd
import numpy as np

# UI Configuration (UI Designer Persona applied)
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Design System matching WCAG AA
st.markdown("""
<style>
    :root {
        --color-primary-500: #3b82f6;
        --color-success: #10b981;
        --color-warning: #f59e0b;
        --color-error: #ef4444;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None

def calc_ci(p, n):
    return 1.96 * np.sqrt((p * (1 - p)) / n)
    
def calc_composite_ci(p, n):
    return 1.96 * np.sqrt(1.5 * (p * (1 - p)) / n)

def main():
    st.title("📊 NLP Sentiment Analysis Dashboard")
    st.markdown("Comparative Evaluation of Transformer vs. Heuristic Models in E-Commerce")
    
    baseline_metrics = load_data('baseline_metrics.json')
    lda_topics = load_data('lda_topics.json')
    transformer_metrics = load_data('transformer_metrics.json')
    
    if not (baseline_metrics and lda_topics and transformer_metrics):
        st.warning("Awaiting metric files generation.")
        return
        
    n_test = baseline_metrics.get('n_test_samples', 2000)
    
    tab1, tab2, tab3 = st.tabs(["🚀 Model Performance", "🔍 Topic Extraction", "ℹ️ Methodology"])
    
    with tab1:
        st.header("Performance Overview")
        
        t_f1 = transformer_metrics['eval_f1']
        t_acc = transformer_metrics['eval_accuracy']
        lr_f1 = baseline_metrics['lr_report']['macro avg']['f1-score']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="DistilBERT F1-Score", value=f"{t_f1:.3f}", delta=f"+{t_f1 - lr_f1:.3f} vs TF-IDF")
        with col2:
            st.metric(label="DistilBERT Accuracy", value=f"{t_acc:.3f}")
        with col3:
            st.metric(label="Evaluation Samples (N)", value=n_test)
            
        st.markdown("---")
        st.subheader("Detailed Macro-Averaged Metrics")
        
        lr_acc = baseline_metrics['lr_accuracy']; lr_p = baseline_metrics['lr_report']['macro avg']['precision']
        lr_r = baseline_metrics['lr_report']['macro avg']['recall']
        nb_acc = baseline_metrics['nb_accuracy']; nb_p = baseline_metrics['nb_report']['macro avg']['precision']
        nb_r = baseline_metrics['nb_report']['macro avg']['recall']; nb_f1 = baseline_metrics['nb_report']['macro avg']['f1-score']
        t_p = transformer_metrics['eval_precision']; t_r = transformer_metrics['eval_recall']
        
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
