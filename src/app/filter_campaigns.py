#!/usr/bin/env python3
"""
Utility functions to filter campaigns based on user donation history.
These functions can be integrated into your recommendation system.
"""

import json
from typing import List, Dict, Any, Set

def load_donations() -> Dict[str, List[Dict[str, Any]]]:
    """Load donation data from JSON file."""
    with open("../../data/donations.json", "r") as f:
        return json.load(f)

def load_campaigns() -> List[Dict[str, Any]]:
    """Load campaign data from JSON file."""
    with open("../../data/campaigns.json", "r") as f:
        return json.load(f)

def get_user_donated_campaign_ids(user_id: str) -> Set[int]:
    """
    Get set of campaign IDs that a user has already donated to.
    
    Args:
        user_id: The user ID to check
        
    Returns:
        Set of campaign IDs the user has donated to
    """
    donations = load_donations()
    donated_campaign_ids = set()
    
    if user_id in donations:
        for donation in donations[user_id]:
            donated_campaign_ids.add(donation["campaign_id"])
    
    return donated_campaign_ids

def get_undonated_campaigns(user_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Get list of campaigns that a user hasn't donated to yet.
    
    Args:
        user_id: The user ID to check
        active_only: If True, only return active campaigns
        
    Returns:
        List of campaign objects that user hasn't donated to
    """
    campaigns = load_campaigns()
    donated_campaign_ids = get_user_donated_campaign_ids(user_id)
    
    undonated_campaigns = []
    for campaign in campaigns:
        # Skip if user already donated to this campaign
        if campaign["id"] in donated_campaign_ids:
            continue
            
        # Skip inactive campaigns if active_only is True
        if active_only and not campaign.get("is_active", True):
            continue
            
        undonated_campaigns.append(campaign)
    
    return undonated_campaigns

def filter_campaigns_by_category(campaigns: List[Dict[str, Any]], category_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Filter campaigns by category IDs.
    
    Args:
        campaigns: List of campaign objects
        category_ids: List of category IDs to filter by
        
    Returns:
        Filtered list of campaigns
    """
    return [c for c in campaigns if c.get("category_id") in category_ids]

def filter_campaigns_by_country(campaigns: List[Dict[str, Any]], country: str) -> List[Dict[str, Any]]:
    """
    Filter campaigns by country.
    
    Args:
        campaigns: List of campaign objects
        country: Country code (e.g., "ID")
        
    Returns:
        Filtered list of campaigns
    """
    return [c for c in campaigns if c.get("country") == country]

def get_recommended_undonated_campaigns(
    user_id: str, 
    category_ids: List[int] = None, 
    country: str = "ID",
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get recommended campaigns that user hasn't donated to, with optional filtering.
    
    Args:
        user_id: The user ID to get recommendations for
        category_ids: Optional list of category IDs to filter by
        country: Country to filter by (default: "ID")
        limit: Maximum number of campaigns to return
        
    Returns:
        List of recommended undonated campaigns
    """
    # Get all undonated campaigns
    undonated = get_undonated_campaigns(user_id, active_only=True)
    
    # Apply filters
    if country:
        undonated = filter_campaigns_by_country(undonated, country)
    
    if category_ids:
        undonated = filter_campaigns_by_category(undonated, category_ids)
    
    # Return limited results
    return undonated[:limit]

# Example usage functions
def example_usage():
    """Example of how to use these functions."""
    
    print("=== EXAMPLE USAGE ===")
    
    # Example 1: Get all undonated campaigns for user 1001
    user_id = "1001"
    undonated = get_undonated_campaigns(user_id)
    print(f"\nUser {user_id} has {len(undonated)} undonated campaigns")
    
    # Example 2: Get undonated health campaigns (category 5)
    health_campaigns = get_recommended_undonated_campaigns(
        user_id=user_id,
        category_ids=[5],  # Health category
        limit=3
    )
    print(f"\nHealth campaigns not donated by user {user_id}:")
    for campaign in health_campaigns:
        print(f"- {campaign['title']} (ID: {campaign['id']})")
    
    # Example 3: Get undonated education campaigns (category 3)
    education_campaigns = get_recommended_undonated_campaigns(
        user_id=user_id,
        category_ids=[3],  # Education category
        limit=3
    )
    print(f"\nEducation campaigns not donated by user {user_id}:")
    for campaign in education_campaigns:
        print(f"- {campaign['title']} (ID: {campaign['id']})")
    
    # Example 4: Check what user has already donated to
    donated_ids = get_user_donated_campaign_ids(user_id)
    print(f"\nUser {user_id} has donated to campaigns: {sorted(list(donated_ids))}")

if __name__ == "__main__":
    example_usage()