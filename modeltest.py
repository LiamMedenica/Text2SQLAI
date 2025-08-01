from transformers import T5Tokenizer, T5ForConditionalGeneration
model_dir = "C:/Users/LiamM/OneDrive/Desktop/Projects/Text2SQLAI/training/checkpoints/t5_finetuned"
try:
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    print("Fine-tuned model loaded successfully")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")