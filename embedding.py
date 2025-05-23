from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.chunks)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def get_context(self, query, top_k=2):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        return "\n".join([self.chunks[i] for i in indices[0]])
