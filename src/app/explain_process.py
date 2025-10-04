#!/usr/bin/env python3
"""
Detailed explanation of the recommendation process.
This script demonstrates how campaigns are recommended based on user's donation history.
"""

import json
import sys
import os
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean import html_to_canonical, template_text
from embeddings import embed_texts
from recommend import weighted_average
from qdrant_store import get_client, query_similar

def load_campaigns() -> List[Dict[str, Any]]:
    """Load campaign data from JSON file."""
    with open("../../data/campaigns.json", "r") as f:
        return json.load(f)

def load_donations() -> Dict[str, List[Dict[str, Any]]]:
    """Load donation data from JSON file."""
    with open("../../data/donations.json", "r") as f:
        return json.load(f)

def explain_recommendation_process(user_id: str, top_k: int = 5):
    """
    Explain the complete recommendation process step by step.
    
    Process:
    1. Load user's donation history
    2. Get last 10 donations for user profiling
    3. Extract campaign descriptions from donated campaigns
    4. Create embeddings from campaign descriptions
    5. Create weighted user profile vector
    6. Search for similar campaigns in vector database
    7. Filter out campaigns user has already donated to
    8. Return top-k recommendations
    """
    
    print("=" * 80)
    print(f"RECOMMENDATION PROCESS EXPLANATION FOR USER {user_id}")
    print("=" * 80)
    
    # Step 1: Load user's donation history
    print("\n📊 STEP 1: Loading user's donation history")
    donations = load_donations()
    all_user_donations = donations.get(user_id, [])
    
    if not all_user_donations:
        print(f"❌ No donations found for user {user_id}")
        return
    
    print(f"✅ Found {len(all_user_donations)} total donations for user {user_id}")
    
    # Get last 10 donations for profiling
    user_dons = all_user_donations[-10:]
    print(f"📈 Using last {len(user_dons)} donations for user profiling")
    
    # Get all donated campaign IDs for filtering
    donated_campaign_ids = set(d["campaign_id"] for d in all_user_donations)
    print(f"🚫 User has donated to {len(donated_campaign_ids)} unique campaigns: {sorted(donated_campaign_ids)}")
    
    # Step 2: Load campaign data
    print("\n📚 STEP 2: Loading campaign data")
    campaigns = {str(c["id"]): c for c in load_campaigns()}
    print(f"✅ Loaded {len(campaigns)} campaigns from database")
    
    # Step 3: Process donated campaigns for user profiling
    print("\n🔍 STEP 3: Processing donated campaigns for user profiling")
    texts = []
    weights = []
    
    print("Processing each donated campaign:")
    for i, d in enumerate(user_dons, 1):
        campaign_id = str(d["campaign_id"])
        amount = d.get("amount", 0)
        
        c = campaigns.get(campaign_id)
        if not c:
            print(f"  {i}. Campaign {campaign_id}: ❌ Not found in database")
            continue
            
        # Clean HTML and create templated text
        canonical = html_to_canonical(c.get("html_description") or "")
        templated = template_text(c["title"], str(c.get("category_id", "")), canonical)
        
        # Calculate weight based on donation amount
        weight = 1.0 + (amount / 100000.0)
        
        texts.append(templated)
        weights.append(weight)
        
        print(f"  {i}. Campaign {campaign_id}: '{c['title']}'")
        print(f"     Amount: IDR {amount:,} → Weight: {weight:.3f}")
        print(f"     Text: {templated[:100]}...")
    
    if not texts:
        print("❌ No valid campaign texts found for user profiling")
        return
    
    # Step 4: Create embeddings
    print(f"\n🧠 STEP 4: Creating embeddings for {len(texts)} campaign texts")
    embs = embed_texts(texts)
    print(f"✅ Generated {len(embs)} embeddings, each with {len(embs[0])} dimensions")
    
    # Step 5: Create weighted user profile vector
    print("\n⚖️ STEP 5: Creating weighted user profile vector")
    user_vec = weighted_average(embs, weights)
    print(f"✅ Created user profile vector with {len(user_vec)} dimensions")
    print(f"📊 Weights used: {[f'{w:.3f}' for w in weights]}")
    
    # Step 6: Search for similar campaigns
    print("\n🔎 STEP 6: Searching for similar campaigns in vector database")
    client = get_client()
    search_limit = max(top_k * 3, 20)  # Get more results to account for filtering
    
    print(f"🎯 Searching for top {search_limit} similar campaigns...")
    print("🔍 Filters applied: is_active=True, country=ID")
    
    res = query_similar(client, user_vec, top_k=search_limit, filters={"is_active": True, "country": "ID"})
    print(f"✅ Found {len(res)} similar campaigns")
    
    # Step 7: Filter out donated campaigns
    print("\n🚫 STEP 7: Filtering out campaigns user has already donated to")
    filtered_recommendations = []
    excluded_count = 0
    
    print("Filtering results:")
    for i, r in enumerate(res, 1):
        payload = r.payload or {}
        campaign_id = payload.get('campaign_id')
        title = payload.get('title', 'Unknown')
        score = r.score
        
        if campaign_id in donated_campaign_ids:
            print(f"  {i}. Campaign {campaign_id}: '{title}' (score: {score:.4f}) → ❌ EXCLUDED (already donated)")
            excluded_count += 1
        else:
            filtered_recommendations.append(r)
            status = "✅ INCLUDED" if len(filtered_recommendations) <= top_k else "⏭️ EXTRA"
            print(f"  {i}. Campaign {campaign_id}: '{title}' (score: {score:.4f}) → {status}")
        
        if len(filtered_recommendations) >= top_k:
            break
    
    print(f"\n📊 Filtering summary:")
    print(f"   - Total candidates: {len(res)}")
    print(f"   - Excluded (already donated): {excluded_count}")
    print(f"   - Final recommendations: {len(filtered_recommendations)}")
    
    # Step 8: Present final recommendations
    print(f"\n🎯 STEP 8: Final recommendations for user {user_id}")
    print("=" * 60)
    
    if not filtered_recommendations:
        print("❌ No new recommendations available (user has donated to all similar campaigns)")
        return
    
    for i, r in enumerate(filtered_recommendations, 1):
        payload = r.payload or {}
        campaign_id = payload.get('campaign_id')
        title = payload.get('title', 'Unknown')
        category_id = payload.get('category_id', 'Unknown')
        score = r.score
        
        print(f"{i:2d}. Campaign ID: {campaign_id}")
        print(f"    Title: {title}")
        print(f"    Category: {category_id}")
        print(f"    Similarity Score: {score:.4f}")
        print()
    
    print("=" * 80)
    print("🎉 RECOMMENDATION PROCESS COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explain the recommendation process")
    parser.add_argument("--user-id", required=True, help="User ID to generate recommendations for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to generate")
    
    args = parser.parse_args()
    explain_recommendation_process(args.user_id, args.top_k)