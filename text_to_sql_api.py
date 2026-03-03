import os
import uuid
import json
import time
import re
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

from search import search_tables, fetch_schema

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing")

client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Enterprise NL2SQL Engine")

QUERY_CACHE: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    session_id: str
    user_query: str
    intent: Dict[str, Any]
    tables_used: List[str]
    sql_query: str


def preprocess_query(query: str) -> str:
    return query.strip()


def extract_intent(user_query: str) -> dict:

    prompt = f"""
You are an enterprise intent extraction engine.

Return STRICT JSON only.
No explanation.
No markdown.
No extra text.

JSON FORMAT:

{{
  "entity": string,
  "metric": string,
  "aggregation": string | null,
  "filters": {{
      "party_name": string | null,
      "year": number | null,
      "month": number | null,
      "location": string | null
  }}
}}

User Query:
"{user_query}"
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 300
        }
    )

    raw_text = response.text.strip()

    # Remove markdown if exists
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    # Try direct JSON load first
    try:
        return json.loads(cleaned)
    except:
        pass

    # Fallback: extract first JSON block safely
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        json_part = cleaned[start:end+1]
        try:
            return json.loads(json_part)
        except:
            pass

    raise HTTPException(
        status_code=500,
        detail=f"Intent extraction failed. Raw output: {raw_text}"
    )

def validate_intent(intent: Dict[str, Any]):

    required_keys = [
        "entity",
        "metric",
        "aggregation",
        "filters"
    ]

    for key in required_keys:
        if key not in intent:
            raise HTTPException(status_code=400, detail="Invalid intent structure")

    return True



def prune_schema(schema: Dict[str, List[Dict[str, Any]]]):

    pruned = {}

    for table, columns in schema.items():

        filtered = []

        for col in columns:
            if col["type"] in [
                "int", "bigint", "decimal",
                "numeric", "float",
                "datetime", "date",
                "varchar", "nvarchar"
            ]:
                filtered.append(col)

        pruned[table] = filtered

    return pruned


def generate_sql(intent: Dict[str, Any],
                 schema: Dict[str, List[Dict[str, Any]]]) -> str:

    prompt = f"""
You are a STRICT Microsoft SQL Server (T-SQL) generator.

Intent:
{json.dumps(intent, indent=2)}

Schema:
{schema}

CRITICAL RULES:
- Use ONLY provided tables.
- Use ONLY provided columns.
- Always use table aliases.
- No SELECT *
- No explanation
- No markdown
- Must end with semicolon
- Must include SELECT and FROM
- SQL Server syntax only

Return SQL only.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 0,
            "max_output_tokens": 600
        }
    )

    sql = response.text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    if not sql.endswith(";"):
        sql += ";"

    return sql


def validate_sql(sql: str,
                 schema: Dict[str, List[Dict[str, Any]]]) -> bool:

    if not sql:
        return False

    sql_upper = sql.upper()

    if "SELECT" not in sql_upper:
        return False

    if "FROM" not in sql_upper:
        return False

    forbidden = ["DROP", "DELETE", "UPDATE", "TRUNCATE", "ALTER"]
    if any(word in sql_upper for word in forbidden):
        return False

    if sql.count("(") != sql.count(")"):
        return False

    valid_tables = list(schema.keys())

    found_tables = re.findall(
        r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)',
        sql,
        re.IGNORECASE
    )

    for table in found_tables:
        if table not in valid_tables:
            return False

    return True

def repair_sql(intent: Dict[str, Any],
               schema: Dict[str, List[Dict[str, Any]]]) -> str:

    prompt = f"""
The previous SQL was invalid.

Intent:
{json.dumps(intent, indent=2)}

Schema:
{schema}

Generate valid SQL.
Return SQL only.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0}
    )

    sql = response.text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    if not sql.endswith(";"):
        sql += ";"

    return sql



@app.post("/ask", response_model=QueryResponse)
def process_query(request: QueryRequest):

    session_id = str(uuid.uuid4())
    user_query = preprocess_query(request.query)

    # Cache check
    if user_query in QUERY_CACHE:
        return QUERY_CACHE[user_query]

    # 1️⃣ Intent
    intent = extract_intent(user_query)
    validate_intent(intent)

    # 2️⃣ FAISS Retrieval
    search_text = user_query + " " + json.dumps(intent)
    predicted_tables, confidence = search_tables(search_text)

    if not predicted_tables:
        raise HTTPException(status_code=404, detail="No relevant tables found")

    table_names = [t for t, _ in predicted_tables]

    schema = fetch_schema(predicted_tables)

    schema = prune_schema(schema)

    sql_query = generate_sql(intent, schema)

    if not validate_sql(sql_query, schema):

        repaired_sql = repair_sql(intent, schema)

        if validate_sql(repaired_sql, schema):
            sql_query = repaired_sql
        else:
            sql_query = "INVALID_QUERY"

    response = QueryResponse(
        session_id=session_id,
        user_query=user_query,
        intent=intent,
        tables_used=table_names,
        sql_query=sql_query
    )

    QUERY_CACHE[user_query] = response

    return response