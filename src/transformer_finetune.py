import torch
import json
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pydantic import BaseModel, Field

from data_loader import load_amazon_subset, DataConfig
import evaluate

class TransformerConfig(BaseModel):
    model_name: str = Field(default="distilbert-base-uncased")
    epochs: int = Field(default=1)
    batch_size: int = Field(default=8)

def finetune_transformer(data: Dict[str, Any], config: TransformerConfig):
    print(f"Fine-tuning {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision="main")
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2, revision="main")
    
    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = data['train'].map(tokenize_function, batched=True)
    tokenized_test = data['test'].map(tokenize_function, batched=True)

    metric_acc = evaluate.load("accuracy")
    metric_p = evaluate.load("precision")
    metric_r = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        prec = metric_p.compute(predictions=predictions, references=labels, average="macro")["precision"]
        rec = metric_r.compute(predictions=predictions, references=labels, average="macro")["recall"]
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Transformer Eval Results: {eval_results}")
    
    with open('transformer_metrics.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

if __name__ == "__main__":
    data_config = DataConfig()
    dataset = load_amazon_subset(data_config)
    trans_config = TransformerConfig()
    
    finetune_transformer(dataset, trans_config)
    print("Transformer fine-tuning complete and metrics saved.")
