import os
from typing import List
from sentence_transformers import SentenceTransformer

_model = None

def get_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(get_model_name())
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    # SentenceTransformers returns list of lists (float32)
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
    return embs
