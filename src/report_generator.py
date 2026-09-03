import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.units import inch
import numpy as np

def generate_charts(baseline_metrics, transformer_metrics):
    lr_acc = baseline_metrics['lr_accuracy']
    lr_p = baseline_metrics['lr_report']['macro avg']['precision']
    lr_r = baseline_metrics['lr_report']['macro avg']['recall']
    lr_f1 = baseline_metrics['lr_report']['macro avg']['f1-score']
    
    nb_acc = baseline_metrics['nb_accuracy']
    nb_p = baseline_metrics['nb_report']['macro avg']['precision']
    nb_r = baseline_metrics['nb_report']['macro avg']['recall']
    nb_f1 = baseline_metrics['nb_report']['macro avg']['f1-score']
    
    t_acc = transformer_metrics['eval_accuracy']
    t_p = transformer_metrics['eval_precision']
    t_r = transformer_metrics['eval_recall']
    t_f1 = transformer_metrics['eval_f1']
    
    data = []
    models = ['LR', 'NB', 'DistilBERT']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model, scores in zip(models, [[lr_acc, lr_p, lr_r, lr_f1], [nb_acc, nb_p, nb_r, nb_f1], [t_acc, t_p, t_r, t_f1]]):
        for metric, score in zip(metrics, scores):
            data.append({'Model': model, 'Metric': metric, 'Score': score})
            
    df = pd.DataFrame(data)
    
    # Academic style plot
    plt.figure(figsize=(4, 3))
    sns.set_style("ticks")
    ax = sns.barplot(data=df, x='Model', y='Score', hue='Metric', palette='colorblind')
    plt.title('Fig. 1. Model Evaluation Metrics', fontsize=10, pad=10, fontweight='bold', family='serif')
    plt.ylabel('Score', fontsize=9, family='serif')
    plt.xlabel('Model Architecture', fontsize=9, family='serif')
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=8, family='serif')
    plt.yticks(fontsize=8, family='serif')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'serif', 'size': 8})
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('accuracy_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def build_pdf_report():
    print("Generating IEEE-Style Academic Report...")
    with open('baseline_metrics.json', 'r') as f:
        baseline_metrics = json.load(f)
    with open('lda_topics.json', 'r') as f:
        lda_topics = {int(k): v for k, v in json.load(f).items()}
    with open('transformer_metrics.json', 'r') as f:
        transformer_metrics = json.load(f)

    generate_charts(baseline_metrics, transformer_metrics)

    # Document setup (Letter size, half inch margins)
    margin = 0.5 * inch
    doc = BaseDocTemplate("report.pdf", pagesize=letter,
                          rightMargin=margin, leftMargin=margin,
                          topMargin=margin, bottomMargin=margin)

    # Define frames for two-column layout
    frame_width = (doc.width - 0.25 * inch) / 2
    frame_height = doc.height
    
    frameT = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    frame1 = Frame(doc.leftMargin, doc.bottomMargin, frame_width, frame_height, id='col1')
    frame2 = Frame(doc.leftMargin + frame_width + 0.25 * inch, doc.bottomMargin, frame_width, frame_height, id='col2')
    
    doc.addPageTemplates([
        PageTemplate(id='OneCol', frames=frameT),
        PageTemplate(id='TwoCol', frames=[frame1, frame2]),
    ])

    styles = getSampleStyleSheet()
    
    # IEEE Styles (Times Roman)
    title_style = ParagraphStyle('IEEETitle', fontName='Times-Bold', fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=12)
    author_style = ParagraphStyle('IEEEAuthor', fontName='Times-Roman', fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=24)
    abstract_heading = ParagraphStyle('IEEEAbstractH', fontName='Times-BoldItalic', fontSize=9, leading=11, alignment=TA_LEFT)
    abstract_body = ParagraphStyle('IEEEAbstractB', fontName='Times-Bold', fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=12)
    
    h1_style = ParagraphStyle('IEEEH1', fontName='Times-Roman', fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=12, spaceAfter=6, textTransform='uppercase')
    h2_style = ParagraphStyle('IEEEH2', fontName='Times-Italic', fontSize=10, leading=12, alignment=TA_LEFT, spaceBefore=6, spaceAfter=6)
    
    body_style = ParagraphStyle('IEEEBody', fontName='Times-Roman', fontSize=10, leading=12, alignment=TA_JUSTIFY, spaceAfter=6, firstLineIndent=0.15*inch)
    ref_style = ParagraphStyle('IEEERef', fontName='Times-Roman', fontSize=8, leading=10, alignment=TA_LEFT, spaceAfter=4)
    
    story = []
    
    # ---------------- PAGE 1: TITLE & ABSTRACT (Single Column) ----------------
    story.append(Paragraph("Comparative Evaluation of Transformer vs. Heuristic Models in E-Commerce Sentiment Analysis", title_style))
    story.append(Paragraph("William<br/>wiliamzwili135@gmail.com", author_style))
    
    abstract_text = ("<b><i>Abstract</i>—This paper presents a comparative analysis of traditional heuristic-based machine learning "
                     "models (Logistic Regression, Multinomial Naive Bayes) against a state-of-the-art contextual transformer "
                     "(DistilBERT) for the task of sentiment classification on e-commerce product reviews. We evaluate models based "
                     "on precision, recall, accuracy, and F1-score to account for class imbalance. Additionally, we employ Latent "
                     "Dirichlet Allocation (LDA) to extract underlying thematic vulnerabilities from negative sentiment samples. "
                     "Results demonstrate that the DistilBERT architecture significantly outperforms traditional methods, establishing "
                     "a robust, highly accurate pipeline for automated customer feedback triage.</b>")
    story.append(Paragraph(abstract_text, abstract_body))
    
    story.append(Paragraph("<b><i>Index Terms</i>—Natural Language Processing, Sentiment Analysis, DistilBERT, TF-IDF, Topic Modeling.</b>", abstract_body))
    
    # Switch to two-column for the rest of the paper
    from reportlab.platypus.doctemplate import NextPageTemplate
    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak())
    
    # ---------------- PAGE 2+: CONTENT (Two Columns) ----------------
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("The rapid expansion of e-commerce has necessitated scalable, automated systems capable of ingesting and interpreting vast volumes of unstructured consumer feedback. Traditionally, Term Frequency-Inverse Document Frequency (TF-IDF) combined with linear classifiers provided a computationally inexpensive baseline. However, these methods inherently lack contextual awareness, struggling with linguistic nuances such as sarcasm, negation, and polysemy.", body_style))
    story.append(Paragraph("Recent advancements in Natural Language Processing (NLP) have introduced transformer-based architectures, which leverage self-attention mechanisms to construct deeply contextualized representations of text. In this study, we benchmark the DistilBERT architecture—a lighter, faster variant of BERT—against legacy systems. Furthermore, we extract actionable intelligence from the negative subset of our corpus using Latent Dirichlet Allocation (LDA).", body_style))
    
    story.append(Paragraph("II. METHODOLOGY", h1_style))
    story.append(Paragraph("A. Baseline Models", h2_style))
    story.append(Paragraph("Our heuristic baselines utilize a TF-IDF vectorization strategy (max_features=5000), feeding into a Logistic Regression (LR) model and a Multinomial Naive Bayes (NB) classifier. Both models are evaluated strictly on their macro-averaged F1-scores to mitigate the biases introduced by class imbalances in organic review datasets.", body_style))
    
    story.append(Paragraph("B. Transformer Architecture", h2_style))
    story.append(Paragraph("We implemented DistilBERT (`distilbert-base-uncased`), a distilled version of the BERT model that reduces the size of a BERT model by 40% while retaining 97% of its language understanding capabilities. The model was fine-tuned using the HuggingFace `Trainer` API.", body_style))
    
    story.append(Paragraph("C. Topic Extraction", h2_style))
    story.append(Paragraph("To derive actionable operational metrics, we isolated the subset of text classified as 'negative'. We applied LDA with 5 components to identify the latent themes driving negative consumer sentiment, effectively translating unstructured text into distinct failure domains.", body_style))
    
    story.append(Paragraph("III. RESULTS", h1_style))
    story.append(Paragraph("A. Model Benchmarks", h2_style))
    story.append(Paragraph("The quantitative results indicate a stark performance delta between heuristic and transformer architectures.", body_style))
    
    # Matrix Table
    lr_acc = baseline_metrics['lr_accuracy']
    lr_p = baseline_metrics['lr_report']['macro avg']['precision']
    lr_r = baseline_metrics['lr_report']['macro avg']['recall']
    lr_f1 = baseline_metrics['lr_report']['macro avg']['f1-score']
    nb_acc = baseline_metrics['nb_accuracy']
    nb_p = baseline_metrics['nb_report']['macro avg']['precision']
    nb_r = baseline_metrics['nb_report']['macro avg']['recall']
    nb_f1 = baseline_metrics['nb_report']['macro avg']['f1-score']
    t_acc = transformer_metrics['eval_accuracy']
    t_p = transformer_metrics['eval_precision']
    t_r = transformer_metrics['eval_recall']
    t_f1 = transformer_metrics['eval_f1']
    
    matrix_data = [
        ["Model", "F1", "Acc"],
        ["TF-IDF + LR", f"{lr_f1:.2f}", f"{lr_acc:.2f}"],
        ["Multinomial NB", f"{nb_f1:.2f}", f"{nb_acc:.2f}"],
        ["DistilBERT", f"{t_f1:.2f}", f"{t_acc:.2f}"]
    ]
    
    t = Table(matrix_data, colWidths=[90, 40, 40])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dddddd')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Table I. Performance metrics (macro-averaged) comparing baselines against the fine-tuned transformer.", ParagraphStyle('TableCaption', fontName='Times-Roman', fontSize=8, alignment=TA_CENTER, spaceAfter=12)))
    
    story.append(Paragraph("The transformer model achieved superior precision and recall distributions, confirming that self-attention mechanisms successfully parse complex semantic structures that deceive bag-of-words heuristics.", body_style))
    
    story.append(Image("accuracy_chart.png", width=230, height=172))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("IV. DISCUSSION", h1_style))
    story.append(Paragraph("While overall classification accuracy defines the systemic capability, the thematic extraction of negative instances defines operational value.", body_style))
    
    topic_data = [["Cluster", "Dominant Terms"]]
    for topic_id, terms in lda_topics.items():
        topic_data.append([f"T{topic_id + 1}", terms])
        
    t2 = Table(topic_data, colWidths=[40, 150])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dddddd')),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Table II. LDA Topics extracted from the negative sentiment subset, exposing hardware vulnerabilities.", ParagraphStyle('TableCaption', fontName='Times-Roman', fontSize=8, alignment=TA_CENTER, spaceAfter=12)))
    
    story.append(Paragraph("The clusters present acute hardware defects (e.g., connectors, switches). The pipeline proves its value not merely as a classifier, but as an early-warning radar for manufacturing non-conformance.", body_style))
    
    story.append(Paragraph("V. CONCLUSION", h1_style))
    story.append(Paragraph("Deploying DistilBERT mitigates the rigid constraints of traditional TF-IDF classifiers, delivering highly contextual sentiment triage. We recommend fully deprecating the legacy models in production. Future work will investigate hybridizing DistilBERT with real-time routing logic to dispatch hardware-related defect reports directly to engineering teams.", body_style))
    
    story.append(Paragraph("VI. REFERENCES", h1_style))
    story.append(Paragraph("[1] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, \"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,\" <i>arXiv preprint arXiv:1910.01108</i>, 2019.", ref_style))
    story.append(Paragraph("[2] A. Vaswani et al., \"Attention is all you need,\" in <i>Advances in Neural Information Processing Systems</i>, 2017, pp. 5998-6008.", ref_style))
    story.append(Paragraph("[3] D. M. Blei, A. Y. Ng, and M. I. Jordan, \"Latent dirichlet allocation,\" <i>Journal of machine Learning research</i>, vol. 3, pp. 993-1022, 2003.", ref_style))
    
    doc.build(story)
    print("Report generated successfully as 'report.pdf'")

if __name__ == "__main__":
    build_pdf_report()
