from embedding import EmbeddingRetriever
from llm_interface import LocalLLM

# Documento base
document_chunks = [
    "I dipendenti possono richiedere ferie con almeno 10 giorni di anticipo.",
    "Le ferie non godute vanno utilizzate entro il 30 giugno dell'anno successivo.",
    "Ogni dipendente ha diritto a 26 giorni lavorativi di ferie all’anno.",
    "I permessi non sono cumulabili con le ferie."
]

# Query
query = "Quanti giorni di ferie ho?"

# Step 1-7: Retrieval
retriever = EmbeddingRetriever(document_chunks)
contexto = retriever.get_context(query)

# Step 8: Prompt
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

# Step 9: LLM risposta
llm = LocalLLM()
risposta = llm.ask(prompt)

print("\nRisposta generata:")
print(risposta)
