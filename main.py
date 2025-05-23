import pandas as pd
from embedding import EmbeddingRetriever
from llm_interface import LocalLLM

# Carica CSV
df = pd.read_csv("./spese.csv")

# Prepara chunk: per ogni riga del CSV crea una stringa tipo
document_chunks = []
for _, row in df.iterrows():
    # esempio: "Housing: Gen 631, Feb 823, Mar 4480, ..., Total 6971"
    # converto i mesi in stringa: 
    monthly = ", ".join([f"{month} {row[month]}" for month in df.columns[1:13]])
    chunk = f"{row['Primary']}: {monthly}, Total: {row['Total']}"
    document_chunks.append(chunk)

# Query
query = "quali sono le total expenses a april?"

# Retrieval
retriever = EmbeddingRetriever(document_chunks)
contexto = retriever.get_context(query)

prompt = f"""
Contesto:
---
{contexto}
---

Domanda:
{query}

Risposta:
"""

print("PROMPT FINALE:")
print(prompt)

llm = LocalLLM()
risposta = llm.ask(prompt)

print("\nRisposta generata:")
print(risposta)
