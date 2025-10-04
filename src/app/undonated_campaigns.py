#!/usr/bin/env python3
"""
Script to find and list campaigns that a user hasn't donated to yet.
This demonstrates the filtering logic used in the recommendation system.
"""

import json
import sys
import os
from typing import List, Dict, Any, Set

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_donations() -> Dict[str, List[Dict[str, Any]]]:
    """Load donation data from JSON file."""
    with open("../../data/donations.json", "r") as f:
        return json.load(f)

def load_campaigns() -> List[Dict[str, Any]]:
    """Load campaign data from JSON file."""
    with open("../../data/campaigns.json", "r") as f:
        return json.load(f)

def get_user_donated_campaigns(user_id: str, donations: Dict[str, List[Dict[str, Any]]]) -> Set[int]:
    """
    Get set of campaign IDs that a user has already donated to.
    
    Args:
        user_id: The user ID to check
        donations: Dictionary with user_id as keys and list of donations as values
        
    Returns:
        Set of campaign IDs the user has donated to
    """
    donated_campaign_ids = set()
    
    if user_id in donations:
        for donation in donations[user_id]:
            donated_campaign_ids.add(donation["campaign_id"])
    
    return donated_campaign_ids

def get_undonated_campaigns(user_id: str) -> List[Dict[str, Any]]:
    """
    Get list of campaigns that a user hasn't donated to yet.
    
    Args:
        user_id: The user ID to check
        
    Returns:
        List of campaign objects that user hasn't donated to
    """
    donations = load_donations()
    campaigns = load_campaigns()
    
    # Get campaigns user has already donated to
    donated_campaign_ids = get_user_donated_campaigns(user_id, donations)
    
    # Filter out donated campaigns
    undonated_campaigns = []
    for campaign in campaigns:
        if campaign["id"] not in donated_campaign_ids:
            undonated_campaigns.append(campaign)
    
    return undonated_campaigns

def display_user_donation_summary(user_id: str):
    """Display summary of user's donation history."""
    donations = load_donations()
    campaigns = load_campaigns()
    
    # Create campaign lookup
    campaign_lookup = {c["id"]: c for c in campaigns}
    
    # Get user's donations
    user_donations = donations.get(user_id, [])
    
    if not user_donations:
        print(f"❌ User {user_id} has no donation history.")
        return
    
    print(f"👤 USER {user_id} DONATION HISTORY")
    print("=" * 50)
    
    total_amount = 0
    donated_campaigns = set()
    
    for donation in user_donations:
        campaign_id = donation["campaign_id"]
        amount = donation["amount"]
        total_amount += amount
        donated_campaigns.add(campaign_id)
        
        campaign_title = "Unknown Campaign"
        if campaign_id in campaign_lookup:
            campaign_title = campaign_lookup[campaign_id]["title"]
        
        print(f"• {campaign_title}")
        print(f"  Campaign ID: {campaign_id}")
        print(f"  Amount: IDR {amount:,}")
        print()
    
    print(f"📊 SUMMARY:")
    print(f"• Total donations: {len(user_donations)}")
    print(f"• Unique campaigns: {len(donated_campaigns)}")
    print(f"• Total amount: IDR {total_amount:,}")
    print()

def display_undonated_campaigns(user_id: str):
    """Display campaigns that user hasn't donated to."""
    undonated = get_undonated_campaigns(user_id)
    
    print(f"🎯 CAMPAIGNS NOT YET DONATED BY USER {user_id}")
    print("=" * 60)
    
    if not undonated:
        print("❌ No undonated campaigns found. User has donated to all available campaigns!")
        return
    
    print(f"Found {len(undonated)} campaigns that user hasn't donated to:\n")
    
    for i, campaign in enumerate(undonated, 1):
        print(f"{i}. {campaign['title']}")
        print(f"   ID: {campaign['id']}")
        print(f"   Category: {campaign['category_id']}")
        print(f"   Country: {campaign['country']}")
        print(f"   Active: {'Yes' if campaign['is_active'] else 'No'}")
        
        # Show brief description
        description = campaign.get('html_description', '')
        if description:
            # Remove HTML tags for brief display
            import re
            clean_desc = re.sub('<[^<]+?>', '', description)
            clean_desc = clean_desc.strip()[:100]
            if len(clean_desc) == 100:
                clean_desc += "..."
            print(f"   Description: {clean_desc}")
        print()

def compare_users(user_id1: str, user_id2: str):
    """Compare donation patterns between two users."""
    print(f"🔄 COMPARING USERS {user_id1} vs {user_id2}")
    print("=" * 60)
    
    donations = load_donations()
    
    user1_campaigns = get_user_donated_campaigns(user_id1, donations)
    user2_campaigns = get_user_donated_campaigns(user_id2, donations)
    
    print(f"User {user_id1} donated to: {len(user1_campaigns)} campaigns")
    print(f"User {user_id2} donated to: {len(user2_campaigns)} campaigns")
    
    # Common campaigns
    common = user1_campaigns.intersection(user2_campaigns)
    print(f"Common campaigns: {len(common)}")
    if common:
        print(f"  Campaign IDs: {sorted(list(common))}")
    
    # Unique to each user
    unique_user1 = user1_campaigns - user2_campaigns
    unique_user2 = user2_campaigns - user1_campaigns
    
    print(f"Unique to User {user_id1}: {len(unique_user1)}")
    if unique_user1:
        print(f"  Campaign IDs: {sorted(list(unique_user1))}")
    
    print(f"Unique to User {user_id2}: {len(unique_user2)}")
    if unique_user2:
        print(f"  Campaign IDs: {sorted(list(unique_user2))}")
    print()

def main():
    """Main function to demonstrate undonated campaign filtering."""
    if len(sys.argv) < 2:
        print("Usage: python undonated_campaigns.py <user_id> [compare_user_id]")
        print("\nExamples:")
        print("  python undonated_campaigns.py 1001")
        print("  python undonated_campaigns.py 2002")
        print("  python undonated_campaigns.py 1001 2002  # Compare two users")
        return
    
    user_id = sys.argv[1]
    
    # Display user's donation history
    display_user_donation_summary(user_id)
    
    # Display undonated campaigns
    display_undonated_campaigns(user_id)
    
    # If second user provided, compare them
    if len(sys.argv) >= 3:
        user_id2 = sys.argv[2]
        compare_users(user_id, user_id2)

if __name__ == "__main__":
    main()