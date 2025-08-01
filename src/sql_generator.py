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
        "AS a t1.id = t2.id" not in sql_query and
        ("COUNT" in sql_query if "count" in query.lower() else True) and
        ("SUM" in sql_query if "total" in query.lower() or "sum" in query.lower() else True) and
        all(col.strip() in schema.get(table, []) or col.strip() == "*" or col.strip().startswith(("SUM(", "COUNT(")) for col in sql_query.split("FROM")[0].replace("SELECT", "").split(",")) and
        (all(any(c["value"].strip("'") in sql_query for c in entities.get("conditions", [])) if entities.get("conditions") else True))
    )

    # Use rule-based fallback if T5 output is invalid or missing conditions
    if not valid_sql or "SELECT * FROM" in sql_query or ("count" in query.lower() and "COUNT" not in sql_query):
        columns = entities.get("columns", ["*"])
        conditions = entities.get("conditions", [])
        intent = entities.get("intent", "select")
        order_by = entities.get("order_by", None)

        # Force columns for aggregate queries
        if "count" in query.lower():
            columns = ["COUNT(*)"]
            intent = "aggregate"
        elif "total" in query.lower() or "sum" in query.lower():
            columns = ["SUM(amount)"]
            intent = "aggregate"
        elif not columns or not all(col in schema.get(table, []) or col == "*" or col.startswith(("SUM(", "COUNT(")) for col in columns):
            columns = entities.get("columns", ["*"])  # Respect extracted columns

        # Handle LIKE conditions
        filtered_conditions = []
        for condition in conditions:
            if condition["operator"] == "LIKE" and condition["column"] in schema.get(table, []):
                value = condition["value"].strip("'")
                condition["value"] = f"'%{value}%'"
                filtered_conditions = [condition]  # Only keep LIKE condition
                break
            elif condition["column"] in schema.get(table, []):
                filtered_conditions.append(condition)

        # Build SQL query
        sql_query = f"SELECT {', '.join(columns)} FROM {table}"
        if filtered_conditions:
            condition_str = " AND ".join([f"{c['column']} {c['operator']} {c['value']}" for c in filtered_conditions])
            sql_query += f" WHERE {condition_str}"
        if order_by and order_by["column"] in schema.get(table, []):
            sql_query += f" ORDER BY {order_by['column']} {order_by['direction'].upper()}"

    return sql_query.strip()