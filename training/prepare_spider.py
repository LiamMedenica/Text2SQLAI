import json
import os
from datasets import load_dataset

def load_spider_schemas():
    """Load schema information from Spider dataset's tables.json."""
    # Download tables.json from Spider dataset
    # Note: You may need to manually download from https://yale-lily.github.io/spider
    # and place it in training/data/tables.json
    tables_file = "C:/Users/LiamM/OneDrive/Desktop/Projects/Text2SQLAI/training/data/tables.json"
    if not os.path.exists(tables_file):
        raise FileNotFoundError("Please download tables.json from https://yale-lily.github.io/spider and place it in training/data/")
    
    with open(tables_file, "r", encoding="utf-8") as f:
        schemas = json.load(f)
    
    schema_dict = {}
    for schema in schemas:
        db_id = schema["db_id"]
        table_names = schema["table_names"]
        column_names = []
        for table_idx, table in enumerate(schema["table_names_original"]):
            for col in schema["column_names_original"]:
                if col[0] == table_idx:
                    column_names.append(f"{table}.{col[1]}")
        schema_dict[db_id] = {
            "table_names": table_names,
            "column_names": column_names
        }
    return schema_dict

def preprocess_spider(dataset, schemas, output_dir):
    """Preprocess Spider dataset for T5 fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)
    
    def format_input(example):
        try:
            question = example["question"]
            db_id = example["db_id"]
            schema = schemas.get(db_id, {"table_names": [], "column_names": []})
            schema_text = f"DB: {db_id} | Tables: {', '.join(schema['table_names'])} | Columns: {', '.join(schema['column_names'])}"
            input_text = f"translate to SQL: {question} | {schema_text}"
            return {"input_text": input_text, "target_text": example["query"]}
        except Exception as e:
            print(f"Error processing example {example}: {e}")
            return None

    # Process train and validation splits
    for split in ["train", "validation"]:
        processed = dataset[split].map(format_input, remove_columns=["question", "query", "db_id"])
        # Filter out None results from errors
        processed = processed.filter(lambda x: x is not None)
        # Save to JSONL
        output_file = os.path.join(output_dir, f"spider_{split}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in processed:
                json.dump({"input": item["input_text"], "target": item["target_text"]}, f)
                f.write("\n")
        print(f"Saved {split} data to {output_file}")

if __name__ == "__main__":
    output_dir = "C:/Users/LiamM/OneDrive/Desktop/Projects/Text2SQLAI/training/data"
    # Load Spider dataset
    spider = load_dataset("spider")
    # Load schemas
    schemas = load_spider_schemas()
    # Preprocess dataset
    preprocess_spider(spider, schemas, output_dir)