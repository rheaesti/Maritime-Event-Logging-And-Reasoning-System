import numpy as np
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    embedding = response.json().get("embedding", [])

    if not embedding:
        raise ValueError("Empty embedding received")

    return np.array(embedding, dtype=np.float32)


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero norm embedding")
    return vec / norm


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add_text(self, text: str):
        if len(text.strip()) < 200:
            return

        emb = normalize(embed_text(text))
        self.vectors.append(emb)
        self.texts.append(text)

    def search(self, query: str, top_k=4):
        query = query.strip()
        if not query:
            raise ValueError("Empty query")

        if not self.vectors:
            raise ValueError("Vector store empty")

        query_vec = normalize(embed_text(query))

        scores = [
            np.dot(query_vec, vector)
            for vector in self.vectors
        ]

        ranked = sorted(
            zip(scores, self.texts),
            key=lambda x: x[0],
            reverse=True
        )

        return [text for _, text in ranked[:top_k]]
