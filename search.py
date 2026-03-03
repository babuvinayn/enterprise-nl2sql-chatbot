import os
import faiss
import pickle
import numpy as np
import json
import pyodbc
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# ============================================================
#                     LOAD ENV VARIABLES
# ============================================================

load_dotenv()

DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


# ============================================================
#                     PATH CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "table_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.pkl")

TOP_TABLES = 5
TOP_K_FAISS = 100


# ============================================================
#                  LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
#                  LOAD FAISS + METADATA
# ============================================================

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

questions = metadata.get("questions", [])
tables = metadata.get("tables", [])


# ============================================================
#                    DB CONNECTION
# ============================================================

def get_connection():
    return pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_SERVER},{DB_PORT};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_USERNAME};"
        f"PWD={DB_PASSWORD};"
        f"TrustServerCertificate=yes;"
    )


# ============================================================
#                   TABLE SEARCH
# ============================================================

def search_tables(user_query, top_k=TOP_K_FAISS):

    query_embedding = model.encode(
        [user_query],
        normalize_embeddings=True
    )[0].astype("float32")

    similarities, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    table_scores = defaultdict(float)

    for score, idx in zip(similarities[0], indices[0]):
        table = tables[idx]
        table_scores[table] = max(table_scores[table], float(score))

    if not table_scores:
        return [], 0.0

    sorted_tables = sorted(
        table_scores.items(),
        key=lambda x: -x[1]
    )

    top_candidates = sorted_tables[:TOP_TABLES]

    scores = np.array([score for _, score in top_candidates], dtype=float)
    total = np.sum(scores)

    if total == 0:
        return [], 0.0

    confidence_scores = scores / total

    top_tables = [
        (table, float(round(confidence_scores[i], 4)))
        for i, (table, _) in enumerate(top_candidates)
    ]

    overall_confidence = float(round(confidence_scores[0], 4))

    return top_tables, overall_confidence


# ============================================================
#                   FETCH SCHEMA
# ============================================================

def fetch_schema(table_list):

    schema = {}
    conn = get_connection()
    cursor = conn.cursor()

    for table, _ in table_list:
        try:
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
            """, table)

            columns = cursor.fetchall()

            schema[table] = [
                {"column": col[0], "type": col[1]}
                for col in columns
            ]

        except Exception:
            schema[table] = []

    conn.close()
    return schema


# ============================================================
#                   COLUMN PREDICTION
# ============================================================

def predict_columns(user_query, schema):

    query_embedding = model.encode(
        [user_query],
        normalize_embeddings=True
    )[0]

    predicted_columns = {}

    for table, columns in schema.items():

        if not columns:
            predicted_columns[table] = []
            continue

        column_scores = []

        for col in columns:
            col_text = f"{col['column']} column of type {col['type']}"
            col_embedding = model.encode(
                [col_text],
                normalize_embeddings=True
            )[0]

            similarity = float(np.dot(query_embedding, col_embedding))
            column_scores.append((col["column"], similarity))

        column_scores.sort(key=lambda x: x[1], reverse=True)

        predicted_columns[table] = [
            col for col, _ in column_scores[:5]
        ]

    return predicted_columns


# ============================================================
#                   LOCAL TEST MODE
# ============================================================

if __name__ == "__main__":

    while True:

        user_input = input("\nEnter your query (or type exit): ")

        if user_input.lower() == "exit":
            break

        predicted_tables, overall_confidence = search_tables(user_input)

        if not predicted_tables:
            print("\nNo strong match found.\n")
            continue

        schema = fetch_schema(predicted_tables)
        relevant_columns = predict_columns(user_input, schema)

        response = {
            "user_query": user_input,
            "predicted_tables": [
                {"table": t, "confidence": c}
                for t, c in predicted_tables
            ],
            "confidence": overall_confidence,
            "relevant_columns": relevant_columns,
            "schema": schema
        }

        print("\n================ JSON OUTPUT ================\n")
        print(json.dumps(response, indent=4))
        print("\n=============================================\n")