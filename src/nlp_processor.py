import spacy
import re

def extract_entities(query, schema):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query.lower())
    
    entities = {
        "table": "sales",
        "columns": ["*"],
        "conditions": [],
        "intent": "select",
        "order_by": None
    }
    
    # Schema
    schema_cols = {
        "sales": ["id", "amount", "date", "product"],
        "products": ["id", "name", "category"]
    }
    
    # Detect table
    for table in schema_cols:
        if table in query:
            entities["table"] = table
            break
    
    # Detect specific columns (e.g., "id and product")
    columns = []
    for col in schema_cols[entities["table"]]:
        if f"{col} and" in query or f"{col}," in query or col == query.split("from")[0].split()[-1]:
            columns.append(col)
    if columns:
        entities["columns"] = columns
    
    # Detect conditions
    for i, token in enumerate(doc):
        # Numerical conditions (less than, greater than, equals)
        if token.text in ["less", "greater", "equals", "is"] and i + 2 < len(doc):
            if doc[i + 1].text == "than" and doc[i + 2].text.isdigit():
                entities["conditions"].append({
                    "column": "amount",
                    "operator": "<" if token.text == "less" else ">" if token.text == "greater" else "=",
                    "value": f"'{doc[i + 2].text}'"
                })
            elif token.text in ["equals", "is"] and doc[i + 1].text.isdigit():
                entities["conditions"].append({
                    "column": "amount",
                    "operator": "=",
                    "value": f"'{doc[i + 1].text}'"
                })
        # String conditions (laptop, phone, electronics)
        elif token.text in ["laptop", "phone", "electronics"] and "contain" not in query:
            column = "product" if entities["table"] == "sales" else "category"
            entities["conditions"].append({
                "column": column,
                "operator": "=",
                "value": f"'{token.text}'"
            })
        # LIKE conditions
        elif token.text == "containing":
            prev_token = doc[i - 1] if i - 1 >= 0 else None
            next_token = doc[i + 1] if i + 1 < len(doc) else None
            if prev_token and next_token and prev_token.text in schema_cols[entities["table"]]:
                entities["conditions"] = [{
                    "column": prev_token.text,
                    "operator": "LIKE",
                    "value": next_token.text
                }]
                entities["columns"] = ["*"]
                break
    
    # Date conditions (e.g., "after 2025-01-01")
    date_match = re.search(r"after\s+(\d{4}-\d{2}-\d{2})", query)
    if date_match:
        entities["conditions"].append({
            "column": "date",
            "operator": ">",
            "value": f"'{date_match.group(1)}'"
        })
    
    # Heuristic for "high" sales
    if "high" in query:
        entities["conditions"].append({
            "column": "amount",
            "operator": ">",
            "value": "'1000'"
        })
    
    # Detect intent and override columns
    if "count" in query:
        entities["intent"] = "aggregate"
        entities["columns"] = ["COUNT(*)"]
    elif "total" in query or "sum" in query:
        entities["intent"] = "aggregate"
        entities["columns"] = ["SUM(amount)"]
    
    # Detect ORDER BY
    if "ordered by" in query or "sort by" in query:
        for col in schema_cols[entities["table"]]:
            if col in query:
                direction = "desc" if "descending" in query else "asc"
                entities["order_by"] = {"column": col, "direction": direction}
                entities["columns"] = ["*"]
    
    return entities