import json
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, accuracy_score
from pydantic import BaseModel, Field

from data_loader import load_amazon_subset, DataConfig

class BaselineConfig(BaseModel):
    max_features: int = Field(default=5000, gt=0)
    lda_n_components: int = Field(default=5, gt=0)
    random_state: int = Field(default=42, ge=0)

def train_sentiment_models(data: Dict[str, Any], config: BaselineConfig) -> Tuple[Dict[str, Any], TfidfVectorizer, Dict[str, Any]]:
    print("Training TF-IDF Baseline Models (Logistic Regression & Naive Bayes)...")
    
    vectorizer = TfidfVectorizer(max_features=config.max_features, stop_words='english')
    
    # Train data
    X_train = vectorizer.fit_transform(data['train']['cleaned_text'])
    y_train = data['train']['label']
    
    # Test data
    X_test = vectorizer.transform(data['test']['cleaned_text'])
    y_test = data['test']['label']
    
    # Model 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=config.random_state)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred, output_dict=True)
    
    # Model 2: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_report = classification_report(y_test, nb_pred, output_dict=True)
    
    print(f"LR Baseline Accuracy: {lr_acc:.4f}")
    print(f"NB Baseline Accuracy: {nb_acc:.4f}")
    
    metrics = {
        'n_test_samples': len(y_test),
        'lr_accuracy': lr_acc,
        'lr_report': lr_report,
        'nb_accuracy': nb_acc,
        'nb_report': nb_report
    }
    
    models = {
        'logistic_regression': lr_model,
        'naive_bayes': nb_model
    }
    
    return models, vectorizer, metrics

def train_topic_model(data: Dict[str, Any], config: BaselineConfig) -> Tuple[LatentDirichletAllocation, CountVectorizer, Dict[int, str]]:
    print(f"Training LDA Topic Model with {config.lda_n_components} components on Negative Reviews...")
    
    # Filter negative reviews from train set (label == 0)
    negative_reviews = data['train'].filter(lambda x: x['label'] == 0)['cleaned_text']
    if not negative_reviews:
        raise ValueError("No negative reviews available for LDA topic modeling.")
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=config.max_features)
    tf = tf_vectorizer.fit_transform(negative_reviews)
    
    lda = LatentDirichletAllocation(n_components=config.lda_n_components, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=config.random_state)
    lda.fit(tf)
    
    # Extract top words per topic
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = " ".join([tf_feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        topics[topic_idx] = top_words
        print(f"Topic #{topic_idx}: {top_words}")
        
    return lda, tf_vectorizer, topics

def save_baseline_artifacts(metrics: Dict[str, Any], topics: Dict[int, str]) -> None:
    with open('baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    with open('lda_topics.json', 'w') as f:
        json.dump(topics, f, indent=4)

if __name__ == "__main__":
    data_config = DataConfig()
    dataset = load_amazon_subset(data_config)
    
    base_config = BaselineConfig()
    
    models, vectorizer, metrics = train_sentiment_models(dataset, base_config)
    lda_model, tf_vectorizer, topics = train_topic_model(dataset, base_config)
    
    save_baseline_artifacts(metrics, topics)
    print("Baseline artifacts saved.")
