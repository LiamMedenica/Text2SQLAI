import spacy
import re
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

def extract_entities(query, schema):
    """Extract entities, columns, and conditions using spaCy and Regex."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query.lower())
    entities = {"table": None, "columns": [], "conditions": [], "intent": "select"}
    logging.debug(f"Processing query: {query}")

    # Regex for numerical conditions (e.g., "amount > 500", "amount is greater than 500")
    number_pattern = re.compile(r"(\w+)\s*(is\s+)?(?:greater than|less than|equals|[><=]=?)\s*(\d+)", re.IGNORECASE)
    number_matches = number_pattern.findall(query)
    logging.debug(f"Numerical matches: {number_matches}")

    # Regex for string conditions (e.g., "product is phone")
    string_pattern = re.compile(r"(\w+)\s*(is|=)\s*([a-zA-Z][a-zA-Z0-9_]*)\b", re.IGNORECASE)
    string_matches = string_pattern.findall(query)
    logging.debug(f"String matches: {string_matches}")

    # Regex for date conditions (e.g., "date is after 2025-01-01")
    date_pattern = re.compile(r"(\w+)\s*(is\s+)?(?:after|before|on)\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
    date_matches = date_pattern.findall(query)
    logging.debug(f"Date matches: {date_matches}")

    # Identify table
    for table in schema["tables"]:
        if table in query:
            entities["table"] = table
            break
    if not entities["table"]:
        entities["table"] = "sales"  # Default table
    logging.debug(f"Selected table: {entities['table']}")

    # Identify columns
    select_all = False
    condition_columns = set()
    query_before_where = query.lower().split("where")[0] if "where" in query.lower() else query.lower()
    
    # Check for explicit columns in query before "where"
    if "get" in query_before_where:
        # Extract potential columns between "get" and "where" or end of query
        column_pattern = re.compile(r"\bget\s+(?:the\s+)?([\w\s,]+?)(?:\s+from|\s+where|$)", re.IGNORECASE)
        match = column_pattern.search(query.lower())
        if match:
            column_text = match.group(1).replace("and", ",").split(",")
            for col in column_text:
                col = col.strip()
                if col in schema["tables"][entities["table"]] and col not in entities["columns"]:
                    entities["columns"].append(col)
                    logging.debug(f"Added column from 'get': {col}")
    
    # Fallback to token-based column detection if no columns found
    if not entities["columns"]:
        for i, token in enumerate(doc):
            if token.text in ["all", "list", "show"]:
                entities["columns"] = ["*"]
                select_all = True
                break
            if token.text in schema["tables"][entities["table"]] and token.text in query_before_where:
                if token.text not in entities["columns"]:
                    entities["columns"].append(token.text)
                    logging.debug(f"Added column: {token.text}")
            # Handle "and" for multiple columns
            if token.text == "and" and i > 0 and i < len(doc) - 1:
                prev_token = doc[i-1].text
                next_token = doc[i+1].text
                if prev_token in schema["tables"][entities["table"]] and prev_token not in entities["columns"]:
                    entities["columns"].append(prev_token)
                    logging.debug(f"Added column via 'and': {prev_token}")
                if next_token in schema["tables"][entities["table"]] and next_token not in entities["columns"]:
                    entities["columns"].append(next_token)
                    logging.debug(f"Added column via 'and': {next_token}")

    if not entities["columns"] and not select_all:
        entities["columns"] = ["*"]
    logging.debug(f"Selected columns: {entities['columns']}")

    # Map natural language operators to SQL operators
    operator_map = {
        "greater than": ">",
        "less than": "<",
        "equals": "=",
        "is": "=",
        "after": ">",
        "before": "<",
        "on": "="
    }

    # Extract numerical conditions
    used_columns = set()
    for match in number_matches:
        column, _, value = match
        if column in schema["tables"][entities["table"]]:
            operator = ">" if "greater than" in query.lower() else "<" if "less than" in query.lower() else "="
            entities["conditions"].append({"column": column, "operator": operator, "value": value})
            used_columns.add(column)
            condition_columns.add(column)
    logging.debug(f"Numerical conditions: {entities['conditions']}")

    # Extract date conditions
    for match in date_matches:
        column, _, value = match
        if column in schema["tables"][entities["table"]]:
            operator = ">" if "after" in query.lower() else "<" if "before" in query.lower() else "="
            entities["conditions"].append({"column": column, "operator": operator, "value": f"'{value}'"})
            used_columns.add(column)
            condition_columns.add(column)
    logging.debug(f"Date conditions: {entities['conditions']}")

    # Extract string conditions (only for unused columns)
    for match in string_matches:
        try:
            column, _, value = match
            if (column in schema["tables"][entities["table"]] and
                column not in used_columns and
                value.lower() not in operator_map and
                value.lower() not in ["greater", "less", "than"]):
                entities["conditions"].append({"column": column, "operator": "=", "value": f"'{value}'"})
                used_columns.add(column)
                condition_columns.add(column)
        except ValueError as e:
            logging.error(f"Error unpacking string match {match}: {e}")
            continue
    logging.debug(f"Final conditions: {entities['conditions']}")

    # Detect aggregation intent
    if "total" in query or "sum" in query:
        entities["intent"] = "aggregate"
        if "amount" in query:
            entities["columns"] = ["SUM(amount)"]
    logging.debug(f"Final entities: {entities}")

    return entities