#!/usr/bin/env python3
"""
Detailed explanation of how similarity calculation works in the recommendation system.
This script demonstrates the complete similarity process from text to recommendations.
"""

import json
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean import html_to_canonical, template_text
from embeddings import embed_texts, get_model_name
from recommend import weighted_average

def load_campaigns() -> List[Dict[str, Any]]:
    """Load campaign data from JSON file."""
    with open("../../data/campaigns.json", "r") as f:
        return json.load(f)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Convert to numpy arrays for easier calculation
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Cosine similarity = dot product / (magnitude of a * magnitude of b)
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def explain_similarity_calculation():
    """
    Explain how similarity is calculated in the recommendation system.
    """
    
    print("=" * 80)
    print("SIMILARITY CALCULATION EXPLANATION")
    print("=" * 80)
    
    # Step 1: Text Processing
    print("\n🔤 STEP 1: TEXT PROCESSING")
    print("-" * 40)
    
    campaigns = load_campaigns()
    
    # Take first two campaigns as examples
    campaign1 = campaigns[0]  # Health clinic
    campaign2 = campaigns[1]  # Education scholarship
    
    print(f"Example Campaign 1: {campaign1['title']}")
    print(f"Category: {campaign1['category_id']}")
    print(f"Raw HTML: {campaign1['html_description'][:100]}...")
    
    # Clean HTML
    canonical1 = html_to_canonical(campaign1['html_description'])
    templated1 = template_text(campaign1['title'], str(campaign1['category_id']), canonical1)
    
    print(f"\nCleaned Text:")
    print(f"{templated1}")
    
    print(f"\nExample Campaign 2: {campaign2['title']}")
    print(f"Category: {campaign2['category_id']}")
    print(f"Raw HTML: {campaign2['html_description'][:100]}...")
    
    # Clean HTML
    canonical2 = html_to_canonical(campaign2['html_description'])
    templated2 = template_text(campaign2['title'], str(campaign2['category_id']), canonical2)
    
    print(f"\nCleaned Text:")
    print(f"{templated2}")
    
    # Step 2: Embedding Model
    print(f"\n🧠 STEP 2: EMBEDDING MODEL")
    print("-" * 40)
    
    model_name = get_model_name()
    print(f"Model: {model_name}")
    print("This is a multilingual sentence transformer that:")
    print("• Converts text into 384-dimensional vectors")
    print("• Understands semantic meaning (not just keywords)")
    print("• Works with Indonesian and English text")
    print("• Normalizes embeddings for cosine similarity")
    
    # Step 3: Vector Embeddings
    print(f"\n📊 STEP 3: VECTOR EMBEDDINGS")
    print("-" * 40)
    
    texts = [templated1, templated2]
    embeddings = embed_texts(texts)
    
    print(f"Text 1 embedding: {len(embeddings[0])} dimensions")
    print(f"First 10 values: {[f'{x:.4f}' for x in embeddings[0][:10]]}")
    print(f"Text 2 embedding: {len(embeddings[1])} dimensions")
    print(f"First 10 values: {[f'{x:.4f}' for x in embeddings[1][:10]]}")
    
    # Step 4: Similarity Calculation
    print(f"\n📐 STEP 4: SIMILARITY CALCULATION")
    print("-" * 40)
    
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    
    print("Similarity Metric: COSINE SIMILARITY")
    print("Formula: cos(θ) = (A · B) / (||A|| × ||B||)")
    print("Where:")
    print("• A · B = dot product of vectors A and B")
    print("• ||A|| = magnitude (length) of vector A")
    print("• ||B|| = magnitude (length) of vector B")
    print()
    print("Range: -1 to 1")
    print("• 1.0 = identical meaning")
    print("• 0.0 = no similarity")
    print("• -1.0 = opposite meaning")
    print()
    print(f"Similarity between '{campaign1['title']}' and '{campaign2['title']}': {similarity:.4f}")
    
    # Step 5: What Affects Similarity
    print(f"\n🎯 STEP 5: WHAT AFFECTS SIMILARITY")
    print("-" * 40)
    
    print("Similarity is based on:")
    print("1. SEMANTIC MEANING:")
    print("   • 'kesehatan' (health) vs 'pendidikan' (education)")
    print("   • 'anak' (children) appears in both → increases similarity")
    print("   • 'bantuan' (help/assistance) → common theme")
    print()
    print("2. CONTEXT UNDERSTANDING:")
    print("   • Model understands 'klinik gratis' ≈ 'layanan kesehatan'")
    print("   • 'beasiswa' ≈ 'bantuan pendidikan'")
    print("   • 'anak-anak' ≈ 'siswa' (both refer to young people)")
    print()
    print("3. CATEGORY INFLUENCE:")
    print("   • Category IDs are included in the text")
    print("   • Same category = higher similarity")
    print("   • Different categories can still be similar if content matches")
    print()
    print("4. DESCRIPTION CONTENT:")
    print("   • Detailed descriptions provide more context")
    print("   • Common keywords increase similarity")
    print("   • Purpose and beneficiaries matter")
    
    # Step 6: User Profile Similarity
    print(f"\n👤 STEP 6: USER PROFILE SIMILARITY")
    print("-" * 40)
    
    print("For recommendations, we create a USER PROFILE VECTOR by:")
    print("1. Taking embeddings of campaigns user donated to")
    print("2. Calculating weighted average based on donation amounts")
    print("3. Using this profile vector to find similar campaigns")
    print()
    print("Example:")
    print("• User donated IDR 100,000 to health campaign")
    print("• User donated IDR 200,000 to education campaign")
    print("• Profile vector = (1.0 × health_vector + 2.0 × education_vector) / 3.0")
    print("• This profile leans more toward education due to higher donation")
    
    # Step 7: Real Examples
    print(f"\n🔍 STEP 7: REAL SIMILARITY EXAMPLES")
    print("-" * 40)
    
    # Calculate similarities between different campaign types
    all_campaigns = campaigns[:5]  # First 5 campaigns
    all_texts = []
    
    for c in all_campaigns:
        canonical = html_to_canonical(c['html_description'])
        templated = template_text(c['title'], str(c['category_id']), canonical)
        all_texts.append(templated)
    
    all_embeddings = embed_texts(all_texts)
    
    print("Similarity matrix (first 5 campaigns):")
    print("Campaigns:")
    for i, c in enumerate(all_campaigns):
        print(f"  {i}: {c['title']} (Category: {c['category_id']})")
    
    print("\nSimilarity scores:")
    for i in range(len(all_campaigns)):
        for j in range(i+1, len(all_campaigns)):
            sim = cosine_similarity(all_embeddings[i], all_embeddings[j])
            print(f"  Campaign {i} ↔ Campaign {j}: {sim:.4f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("• Higher scores = more similar content/purpose")
    print("• Health campaigns tend to cluster together")
    print("• Education campaigns show similarity")
    print("• Emergency/disaster campaigns form their own cluster")
    print("• Cross-category similarity possible (e.g., both help children)")
    print("=" * 80)

if __name__ == "__main__":
    explain_similarity_calculation()