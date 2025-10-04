import argparse
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from qdrant_client.http.models import PointStruct

from .clean import html_to_canonical, template_text
from .embeddings import embed_texts, get_model_name, get_model
from .qdrant_store import (
    get_client,
    ensure_collection,
    upsert_campaigns,
    query_similar,
    get_collection_name,
)
from .recommend import weighted_average

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_campaigns():
    with open(DATA_DIR / "campaigns.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_donations():
    with open(DATA_DIR / "donations.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ingest():
    campaigns = load_campaigns()
    # Build canonical text -> template -> embed
    records = []
    for c in campaigns:
        canonical = html_to_canonical(c.get("html_description") or "")
        templated = template_text(c["title"], str(c.get("category_id", "")), canonical)
        records.append({
            "id": c["id"],  # Use integer ID directly instead of converting to string
            "payload": {
                "campaign_id": c["id"],
                "title": c["title"],
                "category_id": c.get("category_id"),
                "country": c.get("country", "ID"),
                "is_active": bool(c.get("is_active", True)),
            },
            "text": templated
        })

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} campaigns with model: {get_model_name()}")
    embs = embed_texts(texts)
    dim = len(embs[0]) if embs else 384

    client = get_client()
    ensure_collection(client, vector_size=dim)

    points: List[PointStruct] = []
    for rec, vec in zip(records, embs):
        points.append(PointStruct(
            id=rec["id"],
            vector=vec,
            payload=rec["payload"]
        ))

    upsert_campaigns(client, points)
    print(f"Upserted {len(points)} vectors into collection '{get_collection_name()}'")

def recommend(user_id: str, top_k: int = 5):
    donations = load_donations()
    user_dons = donations.get(user_id, [])[-10:]
    if not user_dons:
        print("No donations found for user", user_id)
        return

    # Get all campaign IDs the user has ever donated to (not just last 10)
    all_user_donations = donations.get(user_id, [])
    donated_campaign_ids = set(d["campaign_id"] for d in all_user_donations)
    print(f"User {user_id} has donated to {len(donated_campaign_ids)} campaigns: {sorted(donated_campaign_ids)}")

    # For demo, get the embeddings of donated campaigns by re-embedding their text again.
    # In production, you'd either fetch from vector store or cache.
    campaigns = {str(c["id"]): c for c in load_campaigns()}
    texts = []
    weights = []
    for d in user_dons:
        c = campaigns.get(str(d["campaign_id"]))
        if not c:
            continue
        canonical = html_to_canonical(c.get("html_description") or "")
        templated = template_text(c["title"], str(c.get("category_id", "")), canonical)
        texts.append(templated)
        # simple weight by amount (log), you can add recency decay
        amt = max(1.0, float(d.get("amount", 1)))
        weights.append(1.0 + (amt/100000.0))  # crude weight

    if not texts:
        print("No text available for user profile")
        return

    embs = embed_texts(texts)
    user_vec = weighted_average(embs, weights)

    client = get_client()
    # Get more results initially to filter out donated campaigns
    search_limit = max(top_k * 3, 20)  # Get 3x more results to account for filtering
    res = query_similar(client, user_vec, top_k=search_limit, filters={"is_active": True, "country": "ID"})

    # Filter out campaigns the user has already donated to
    filtered_recommendations = []
    for r in res:
        payload = r.payload or {}
        campaign_id = payload.get('campaign_id')
        if campaign_id not in donated_campaign_ids:
            filtered_recommendations.append(r)
        if len(filtered_recommendations) >= top_k:
            break

    print(f"\nTop {len(filtered_recommendations)} NEW recommendations for user {user_id}:")
    print("(Excluding campaigns you've already donated to)")
    for i, r in enumerate(filtered_recommendations, 1):
        payload = r.payload or {}
        print(f"{i:2d}. id={payload.get('campaign_id')} title={payload.get('title')} score={r.score:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Qdrant demo")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ingest", help="Ingest sample campaigns into Qdrant")

    rec = sub.add_parser("recommend", help="Get recommendations for a user")
    rec.add_argument("--user-id", required=True)
    rec.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "recommend":
        recommend(args.user_id, top_k=args.top_k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
