from flask import Flask, request, jsonify, render_template
from src.nlp_processor import extract_entities
from src.sql_generator import generate_sql
from src.schema import load_schema
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

app = Flask(__name__)

# Load schema
schema = load_schema()

# Load fine-tuned T5 model globally
model_dir = "C:/Users/LiamM/OneDrive/Desktop/Projects/Text2SQLAI/training/checkpoints/t5_finetuned"
try:
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
except:
    print("Falling back to pre-trained t5-small model")
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/text2sql', methods=['POST'])
def text2sql():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    entities = extract_entities(query, schema)
    sql_query = generate_sql(query, entities, tokenizer, model)
    return jsonify({"sql": sql_query})

if __name__ == '__main__':
    app.run(debug=True)