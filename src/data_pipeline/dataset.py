import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer

class SecurityTextDataset(Dataset):
    def __init__(self, model_id: str, max_length: int = 512):
        """
        Loads a security/vulnerability dataset and prepares it for distributed training.
        """
        print("Initializing tokenizer and downloading dataset...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Modern open-source LLMs (like Llama or EleutherAI models) often don't have a default pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length

        # Loading a clean, open-source security instruction dataset
        # This dataset contains security rules, exploits, and mitigation text pairs
        self.dataset = load_dataset("NebulaAlpha/CyberSecEval", split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Pull raw text from the dataset (adjust keys based on dataset schema)
        item = self.dataset[idx]
        
        # Combine prompt/instruction and response text if available
        # Adjusting for a robust fallback text structure
        text = item.get("text", "") if "text" in item else f"Context: {item.get('input', '')} Response: {item.get('output', '')}"

        # Tokenize inputs for causal language modeling (CLM)
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Squeeze out the artificial batch dimension added by return_tensors="pt"
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # For Causal LM, labels are exactly identical to input_ids.
        # The model automatically handles internal shifting during loss calculation.
        labels = input_ids.clone()
        
        # Mask out padding tokens from loss calculation so the model doesn't optimize for predicting spaces
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def get_distributed_dataloader(model_id: str, batch_size_per_gpu: int, max_length: int = 512) -> DataLoader:
    """
    Constructs a distributed-safe DataLoader by wrapping the dataset with a DistributedSampler.
    """
    dataset = SecurityTextDataset(model_id=model_id, max_length=max_length)
    
    # Crucial step for Multi-GPU training: The sampler chunks the data so 
    # GPU_0 gets a completely different subset than GPU_1.
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        seed=42,
        drop_last=True  # Guarantees even batch distribution across your GPUs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        pin_memory=True,  # Speeds up tensor transfer from host CPU memory to GPU VRAM
        num_workers=2     # Parallelizes data preprocessing threads
    )
    
    return dataloader

# Quick structural verification test
if __name__ == "__main__":
    # Test using a lightweight open-source base architecture tokenizer (GPT-2 or Pythia)
    TEST_MODEL = "EleutherAI/pythia-70m"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    loader = get_distributed_dataloader(model_id=TEST_MODEL, batch_size_per_gpu=2)
    sample_batch = next(iter(loader))
    print("\n--- Pipeline Data Check Successful ---")
    print(f"Batch Input IDs Shape:     {sample_batch['input_ids'].shape}")
    print(f"Batch Attention Mask Shape: {sample_batch['attention_mask'].shape}")
    print(f"Batch Labels Shape:         {sample_batch['labels'].shape}")