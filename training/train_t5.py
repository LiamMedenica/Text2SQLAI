import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import os

class SpiderDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["target"]

        inputs = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

def main():
    # Paths
    data_dir = "C:/Users/LiamM/OneDrive/Desktop/Projects/Text2SQLAI/training/data"
    train_file = os.path.join(data_dir, "spider_train.jsonl")
    val_file = os.path.join(data_dir, "spider_validation.jsonl")
    output_dir = "C:/Users/LiamM/OneDrive/Desktop/Projects/Text2SQLAI/training/checkpoints"

    # Verify data files exist
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Training or validation file not found: {train_file}, {val_file}")

    # Load tokenizer and model
    model_name = "t5-small"
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # Use new tokenizer behavior
    except ImportError:
        print("Falling back to legacy tokenizer due to missing protobuf")
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load datasets
    train_dataset = SpiderDataset(train_file, tokenizer)
    val_dataset = SpiderDataset(val_file, tokenizer)

    # Training arguments
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Set for testing
            per_device_train_batch_size=4,  # Suitable for 16GB RAM
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            eval_strategy="steps",  # For transformers>=4.36.0
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
        )
    except TypeError:
        print("Falling back to evaluation_strategy for older transformers version")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            evaluation_strategy="steps",  # For transformers<4.36.0
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(output_dir, "t5_finetuned"))
    tokenizer.save_pretrained(os.path.join(output_dir, "t5_finetuned"))
    print("Model saved to", os.path.join(output_dir, "t5_finetuned"))

if __name__ == "__main__":
    main()