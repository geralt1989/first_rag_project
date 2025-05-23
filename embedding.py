import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingRetriever:
    def __init__(self, document_chunks):
        self.chunks = document_chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.chunks)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        # Crea stringhe uniche per ogni riga combinando le colonne (es: categoria e mesi)
        chunks = []
        for _, row in df.iterrows():
            if pd.notna(row['Primary']):
                voce = row['Primary']
                spese = ", ".join(f"{mese}: {row[mese]}" for mese in df.columns[1:13] if pd.notna(row[mese]))
                chunks.append(f"{voce} - {spese}")
        return cls(chunks)

    def get_context(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        return "\n".join([self.chunks[i] for i in indices[0]])
