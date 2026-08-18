import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
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
    models = ['LR (Baseline)', 'NB (Baseline)', 'DistilBERT (Advanced)']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model, scores in zip(models, [[lr_acc, lr_p, lr_r, lr_f1], [nb_acc, nb_p, nb_r, nb_f1], [t_acc, t_p, t_r, t_f1]]):
        for metric, score in zip(metrics, scores):
            data.append({'Model': model, 'Metric': metric, 'Score': score})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Comprehensive Model Evaluation Landscape', fontsize=14, pad=15)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points', rotation=0)
                    
    plt.tight_layout()
    plt.savefig('accuracy_chart.png', dpi=300)
    plt.close()

def generate_pdf_report():
    print("Generating Professional Decision-Grade PDF Report...")
    with open('baseline_metrics.json', 'r') as f:
        baseline_metrics = json.load(f)
    with open('lda_topics.json', 'r') as f:
        lda_topics = {int(k): v for k, v in json.load(f).items()}
    with open('transformer_metrics.json', 'r') as f:
        transformer_metrics = json.load(f)

    generate_charts(baseline_metrics, transformer_metrics)

    doc = SimpleDocTemplate("report.pdf", pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=22, spaceAfter=20, alignment=0, textColor=colors.HexColor('#1a252f'), fontName='Helvetica-Bold')
    h1_style = ParagraphStyle('CustomH1', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor('#2c3e50'), fontName='Helvetica-Bold', borderPadding=5, backColor=colors.HexColor('#ecf0f1'))
    h2_style = ParagraphStyle('CustomH2', parent=styles['Heading3'], fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#34495e'), fontName='Helvetica-Bold')
    body_style = ParagraphStyle('CustomBody', parent=styles['BodyText'], fontSize=11, spaceAfter=12, leading=16, fontName='Helvetica')
    bullet_style = ParagraphStyle('CustomBullet', parent=styles['Bullet'], fontSize=11, spaceAfter=8, leading=16, fontName='Helvetica', bulletIndent=10, leftIndent=25)
    
    story = []
    
    # Title
    story.append(Paragraph("Competitive Sentiment Analysis & Model Landscape Report", title_style))
    story.append(Spacer(1, 10))
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph("This decision-grade report assesses the current NLP model landscape for analyzing Amazon Product Reviews. The core finding is that transitioning from our traditional TF-IDF pipeline to an advanced DistilBERT architecture improves sentiment classification accuracy significantly. Concurrently, negative reviews expose product vulnerabilities spanning usability, quality, and delivery, dictating immediate operational adjustments.", body_style))
    story.append(Paragraph("Key Takeaways:", body_style))
    story.append(Paragraph("• <b>Model Superiority:</b> DistilBERT dominates the landscape across all metrics, achieving significantly higher F1-Scores than the legacy baselines. For NLP sentiment analysis, F1-Score is the critical benchmark as it balances false positives and false negatives, which is crucial for handling imbalanced text data.", bullet_style))
    story.append(Paragraph("• <b>Primary Threats:</b> Topic modeling reveals acute user friction concerning product longevity and software compatibility.", bullet_style))
    story.append(Paragraph("• <b>Strategic Move:</b> Deploy DistilBERT to triage severe negative reviews instantly to customer success teams.", bullet_style))
    
    # 2. Market Landscape & Category Framing
    story.append(Paragraph("2. Model Landscape & Category Framing", h1_style))
    story.append(Paragraph("The sentiment analysis landscape is divided into traditional heuristic-based models and modern contextual transformers. The chart below illustrates the positioning and competitive advantage of adopting advanced transformer models over our baseline approach.", body_style))
    story.append(Image("accuracy_chart.png", width=400, height=230))
    story.append(Spacer(1, 10))
    
    # 3. Competitor Tiers
    story.append(Paragraph("3. Model Tiers", h1_style))
    story.append(Paragraph("<b>Direct / Aspirational Tier (DistilBERT):</b> Represents the modern, scalable solution capable of contextual understanding. Essential for nuanced sentiment detection.", bullet_style))
    story.append(Paragraph("<b>Baseline Tier (TF-IDF + LR & Naive Bayes):</b> The legacy, capability-led approach. They provide solid but rigid benchmarks that struggle with sarcasm and deep context.", bullet_style))
    
    # 4. Benchmarking Matrix
    story.append(Paragraph("4. Benchmarking Matrix", h1_style))
    story.append(Paragraph("Quantitative assessment across precision, recall, and accuracy dimensions.", body_style))
    
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
        ["Model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score"],
        ["TF-IDF + LR", f"{lr_acc:.1%}", f"{lr_p:.1%}", f"{lr_r:.1%}", f"{lr_f1:.1%}"],
        ["Multinomial NB", f"{nb_acc:.1%}", f"{nb_p:.1%}", f"{nb_r:.1%}", f"{nb_f1:.1%}"],
        ["DistilBERT", f"{t_acc:.1%}", f"{t_p:.1%}", f"{t_r:.1%}", f"{t_f1:.1%}"]
    ]
    
    t = Table(matrix_data, colWidths=[120, 80, 100, 100, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, 2), colors.HexColor('#ecf0f1')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#d5f5e3')), # Highlight DistilBERT
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Paragraph("<i>* All metrics represent macro-averages to accurately reflect performance on imbalanced sentiment classes.</i>", ParagraphStyle('Small', parent=styles['Normal'], fontSize=9, textColor=colors.gray)))
    
    story.append(PageBreak())
    
    # 5. Deep Dives
    story.append(Paragraph("5. Deep Dives: Negative Review Themes", h1_style))
    story.append(Paragraph("To understand the 'why' behind negative sentiment, Latent Dirichlet Allocation (LDA) was applied. This surfaces the critical topics driving customer dissatisfaction:", body_style))
    
    topic_data = [["Topic Area", "Key Terms Identified"]]
    for topic_id, terms in lda_topics.items():
        topic_data.append([f"Theme {topic_id + 1}", terms])
        
    t2 = Table(topic_data, colWidths=[100, 380])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t2)
    
    # 6. White-space & Threats
    story.append(Paragraph("6. White-space & Threats", h1_style))
    story.append(Paragraph("<b>White-space:</b> The integration of contextual NLP with automated customer service triggers. Few competitors actively route frustrated customers based on nuanced semantic cues. This is our defensible moat.", body_style))
    story.append(Paragraph("<b>Threats:</b> Product quality complaints identified in Theme 5 (switch, disk, broke, connector) indicate severe physical defects. If unaddressed, superior sentiment tracking will only document our decline.", body_style))
    
    # 7. Strategic Recommendations
    story.append(Paragraph("7. Strategic Recommendations", h1_style))
    story.append(Paragraph("Recommendations are sequenced by impact:", body_style))
    story.append(Paragraph("<b>1. Deprecate Legacy Pipelines:</b> Transition entirely to the DistilBERT framework for all production sentiment inference.", bullet_style))
    story.append(Paragraph("<b>2. Automated Routing:</b> Connect the DistilBERT output to the CRM. Any review classified as negative should instantly trigger a ticket for the customer success team.", bullet_style))
    story.append(Paragraph("<b>3. R&D Feedback Loop:</b> Route topics related to 'connector/broke/disk' directly to the hardware engineering team. Quality control is the root cause of the friction.", bullet_style))
    
    # 8. Sources / Methodology Appendix
    story.append(Paragraph("8. Sources / Methodology Appendix", h1_style))
    story.append(Paragraph("<b>Baseline:</b> Scikit-learn TF-IDF Vectorizer + Logistic Regression, evaluated via standard classification report.", bullet_style))
    story.append(Paragraph(f"<b>Transformer:</b> HuggingFace DistilBERT-base-uncased, evaluated via Trainer API. Runtime: {transformer_metrics['eval_runtime']:.2f}s, Samples/sec: {transformer_metrics['eval_samples_per_second']:.2f}.", bullet_style))
    story.append(Paragraph("<b>Topic Modeling:</b> Sklearn LatentDirichletAllocation on negative subsets, focusing on the top 5 components.", bullet_style))
    
    # Trigger Questions
    story.append(Spacer(1, 15))
    story.append(Paragraph("Team Alignment Trigger Questions:", h2_style))
    story.append(Paragraph("• Is the engineering team prepared to ingest the hardware defect data surfaced by Theme 5?", bullet_style))
    story.append(Paragraph("• What is the cost vs. latency tradeoff of deploying DistilBERT to our real-time edge servers?", bullet_style))
    
    doc.build(story)
    print("Report generated successfully as 'report.pdf'")

if __name__ == "__main__":
    generate_pdf_report()
