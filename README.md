# Text2SQLAI
A Flask-based web application that converts natural language queries into SQL using a fine-tuned T5 model and rule-based logic.

## Overview
Text2SQLAI translates user queries (e.g., "List sales where amount is less than 200") into SQL queries for a predefined database schema. It uses a fine-tuned T5 model (`t5-small`) trained on the Spider dataset, with a rule-based fallback for robustness.

## Features
- Converts natural language to SQL queries.
- Supports queries on `sales` and `products` tables.
- Fine-tuned T5 model with rule-based fallback.
- Flask web interface for easy interaction.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LiamMedenica/Text2SQLAI.git
   cd Text2SQLAI

2. Create and activate a virtual environment:
   python -m venv venv
   .\venv\Scripts\activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Run the Flask App:
   python app.py

## Usage
- Enter a query like "Show sales where product is laptop" in the web interface.
- The app returns SQL, e.g., SELECT * FROM sales WHERE product = 'laptop'.

## Project Structure
- app.py: Flask application.
- src/: Contains nlp_processor.py, sql_generator.py, and schema.py.
- training/: Scripts for data preprocessing (prepare_spider.py) and model training (train_t5.py).
- templates/: HTML templates for the web interface.
- requirements.txt: Python dependencies.

## Requriements 
- Python 3.8+
- Dependencies listed in requirements.txt

## Training
python training/train_t5.py

## Future Improvements / Goals
- Add support for JOIN queries.
- Enable dynamic schema uploads.
- Improve handling of ambiguous queries (e.g., "Find high sales").