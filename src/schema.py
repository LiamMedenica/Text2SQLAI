import json
import os

def load_schema():
    """Load database schema from config/schema.json."""
    schema_path = os.path.join(os.path.dirname(__file__), '../config/schema.json')
    with open(schema_path, 'r') as f:
        return json.load(f)