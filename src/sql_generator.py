def generate_sql(query, entities, tokenizer, model):
    """Generate SQL query using fine-tuned T5 model with robust rule-based fallback."""
    # Format input with schema
    table = entities.get("table", "sales")
    schema = {
        "sales": ["id", "amount", "date", "product"],
        "products": ["id", "name", "category"]
    }
    schema_text = f"DB: default | Tables: {', '.join(schema.keys())} | Columns: {', '.join([f'{table}.{col}' for table, cols in schema.items() for col in cols])}"
    input_text = f"translate to SQL: {query} | {schema_text}"

    # Generate SQL with T5
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Validate T5 output
    valid_sql = (
        sql_query.startswith("SELECT") and
        "FROM" in sql_query and
        "SELECT SELECT" not in sql_query and
        "AS a t1.id = t2.id" not in sql_query and  # Catch malformed joins
        all(col in schema.get(table, []) or col == "*" or col.startswith("SUM(") for col in sql_query.split("FROM")[0].replace("SELECT", "").split(","))
    )

    # Use rule-based fallback if T5 output is invalid
    if not valid_sql:
        columns = entities.get("columns", ["*"])
        conditions = entities.get("conditions", [])

        # Validate and adjust columns
        if not columns or not all(col in schema.get(table, []) or col == "*" or col.startswith("SUM(") for col in columns):
            columns = ["*"] if entities["intent"] != "aggregate" else ["SUM(amount)"]

        # Validate and adjust conditions
        conditions = [c for c in conditions if c["column"] in schema.get(table, [])]

        # Build SQL query
        sql_query = f"SELECT {', '.join(columns)} FROM {table}"
        if conditions:
            condition_str = " AND ".join([f"{c['column']} {c['operator']} {c['value']}" for c in conditions])
            sql_query += f" WHERE {condition_str}"

    return sql_query.strip()