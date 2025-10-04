import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

def get_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)

def get_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "campaigns_demo")

def ensure_collection(client: QdrantClient, vector_size: int):
    name = get_collection_name()
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def upsert_campaigns(client: QdrantClient, points: List[PointStruct]):
    client.upsert(collection_name=get_collection_name(), points=points)

def query_similar(
    client: QdrantClient,
    vector: List[float],
    top_k: int = 50,
    filters: Optional[Dict[str, Any]] = None,
):
    qfilter = None
    if filters:
        conditions = []
        for k, v in filters.items():
            conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        qfilter = Filter(should=conditions)
    return client.search(
        collection_name=get_collection_name(),
        query_vector=vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True
    )
