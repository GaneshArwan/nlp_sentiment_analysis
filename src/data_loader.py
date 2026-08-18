import re
from typing import Dict, List, Any
from datasets import load_dataset
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    dataset_name: str = Field(default="fancyzhx/amazon_polarity")
    sample_size: int = Field(default=2000, description="Size of the subset to load for quicker processing")
    seed: int = Field(default=42)

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_amazon_subset(config: DataConfig) -> Dict[str, Any]:
    """
    Loads a subset of the Amazon Polarity dataset.
    Returns a dictionary with 'train' and 'test' sets.
    """
    print(f"Loading {config.dataset_name} dataset...")
    # Loading a small subset using the split+select syntax
    dataset = load_dataset(config.dataset_name, revision="main")  # nosec B615
    
    # Stratified or random sampling can be done, here we just shuffle and select
    train_subset = dataset['train'].shuffle(seed=config.seed).select(range(config.sample_size))
    test_subset = dataset['test'].shuffle(seed=config.seed).select(range(int(config.sample_size * 0.2)))
    
    # Process text
    def process_batch(examples):
        # amazon_polarity has 'content' and 'label' (0 for negative, 1 for positive)
        cleaned = [clean_text(doc) for doc in examples['content']]
        return {'cleaned_text': cleaned}
    
    print("Cleaning text data...")
    train_subset = train_subset.map(process_batch, batched=True)
    test_subset = test_subset.map(process_batch, batched=True)
    
    return {
        'train': train_subset,
        'test': test_subset
    }

if __name__ == "__main__":
    config = DataConfig()
    data = load_amazon_subset(config)
    print(f"Train size: {len(data['train'])}")
    print(f"Test size: {len(data['test'])}")
    print("Sample positive review:", data['train'].filter(lambda x: x['label'] == 1)['cleaned_text'][0])
    print("Sample negative review:", data['train'].filter(lambda x: x['label'] == 0)['cleaned_text'][0])
