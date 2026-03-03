import os
import pandas as pd
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

all_questions = []
all_tables = []

print("Reading Excel files...")

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".xlsx"):
        table_name = file.replace(".xlsx", "")
        file_path = os.path.join(DATA_FOLDER, file)

        df = pd.read_excel(file_path, header=None)

        for q in df[0].dropna():
            question = str(q).strip()
            if question:
                enriched = f"Table: {table_name} | Question: {question}"
                all_questions.append(enriched)
                all_tables.append(table_name)

print(f"Total questions indexed: {len(all_questions)}")

print("Generating embeddings...")

embeddings = model.encode(
    all_questions,
    batch_size=128,
    show_progress_bar=True,
    normalize_embeddings=True   # IMPORTANT
)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine
index.add(embeddings)

print("Saving FAISS index...")

faiss.write_index(index, "table_index.faiss")

with open("metadata.pkl", "wb") as f:
    pickle.dump({
        "questions": all_questions,
        "tables": all_tables
    }, f)

print("Index built successfully.")