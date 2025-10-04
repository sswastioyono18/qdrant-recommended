"""
Enhanced Donation Analysis Module

This module provides comprehensive analysis of user donation patterns to improve
recommendation quality for undonated campaigns.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics


class DonationAnalyzer:
    """Analyzes user donation patterns to extract insights for better recommendations."""
    
    def __init__(self, donations_path: str = "../../data/donations.json", 
                 projects_path: str = "../../data/projects.json"):
        """Initialize the analyzer with data paths."""
        self.donations_path = donations_path
        self.projects_path = projects_path
        self.donations = self._load_donations()
        self.projects = self._load_projects()
        self.project_lookup = {str(p['id']): p for p in self.projects}
    
    def _load_donations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load donations data from JSON file."""
        with open(self.donations_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_projects(self) -> List[Dict[str, Any]]:
        """Load projects data from JSON file."""
        with open(self.projects_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_user_donation_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Create comprehensive donation profile for a user.
        
        Returns detailed analysis including:
        - Donation frequency and patterns
        - Category preferences with weights
        - Amount patterns and generosity level
        - Temporal patterns (when they donate)
        - Project type preferences
        """
        user_donations = self.donations.get(user_id, [])
        
        if not user_donations:
            return {"error": f"No donations found for user {user_id}"}
        
        # Basic statistics
        total_donations = len(user_donations)
        total_amount = sum(d['amount'] for d in user_donations)
        avg_amount = total_amount / total_donations
        
        # Get project details for each donation
        enriched_donations = []
        for donation in user_donations:
            project_id = str(donation['campaign_id'])
            project = self.project_lookup.get(project_id, {})
            enriched_donations.append({
                **donation,
                'project': project
            })
        
        # Category analysis
        category_stats = self._analyze_categories(enriched_donations)
        
        # Amount analysis
        amount_stats = self._analyze_amounts(user_donations)
        
        # Temporal analysis
        temporal_stats = self._analyze_temporal_patterns(user_donations)
        
        # Project characteristics analysis
        project_stats = self._analyze_project_characteristics(enriched_donations)
        
        return {
            "user_id": user_id,
            "basic_stats": {
                "total_donations": total_donations,
                "total_amount": total_amount,
                "average_amount": avg_amount,
                "unique_projects": len(set(d['campaign_id'] for d in user_donations))
            },
            "category_preferences": category_stats,
            "amount_patterns": amount_stats,
            "temporal_patterns": temporal_stats,
            "project_characteristics": project_stats,
            "generosity_score": self._calculate_generosity_score(user_donations),
            "diversity_score": self._calculate_diversity_score(enriched_donations)
        }
    
    def _analyze_categories(self, enriched_donations: List[Dict]) -> Dict[str, Any]:
        """Analyze user's category preferences with detailed insights."""
        category_counts = Counter()
        category_amounts = defaultdict(list)
        
        for donation in enriched_donations:
            project = donation.get('project', {})
            category_id = project.get('category_id')
            category_name = project.get('category_name', 'Unknown')
            amount = donation['amount']
            
            if category_id:
                category_key = f"{category_name} (ID: {category_id})"
                category_counts[category_key] += 1
                category_amounts[category_key].append(amount)
        
        # Calculate category preferences with weights
        total_donations = len(enriched_donations)
        total_amount = sum(d['amount'] for d in enriched_donations)
        
        category_preferences = {}
        for category, count in category_counts.items():
            amounts = category_amounts[category]
            category_total = sum(amounts)
            
            category_preferences[category] = {
                "donation_count": count,
                "frequency_weight": count / total_donations,
                "total_amount": category_total,
                "amount_weight": category_total / total_amount,
                "avg_amount": statistics.mean(amounts),
                "combined_score": (count / total_donations) * 0.6 + (category_total / total_amount) * 0.4
            }
        
        # Sort by combined score
        sorted_preferences = dict(sorted(
            category_preferences.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        ))
        
        return {
            "preferences": sorted_preferences,
            "top_category": max(category_preferences.keys(), key=lambda k: category_preferences[k]['combined_score']) if category_preferences else None,
            "category_diversity": len(category_preferences)
        }
    
    def _analyze_amounts(self, donations: List[Dict]) -> Dict[str, Any]:
        """Analyze donation amount patterns."""
        amounts = [d['amount'] for d in donations]
        
        return {
            "min_amount": min(amounts),
            "max_amount": max(amounts),
            "median_amount": statistics.median(amounts),
            "std_deviation": statistics.stdev(amounts) if len(amounts) > 1 else 0,
            "amount_ranges": self._categorize_amounts(amounts),
            "consistency_score": 1 / (1 + statistics.stdev(amounts) / statistics.mean(amounts)) if len(amounts) > 1 else 1
        }
    
    def _categorize_amounts(self, amounts: List[float]) -> Dict[str, int]:
        """Categorize donation amounts into ranges."""
        ranges = {
            "small (< 50k)": 0,
            "medium (50k-200k)": 0,
            "large (200k-500k)": 0,
            "very_large (> 500k)": 0
        }
        
        for amount in amounts:
            if amount < 50000:
                ranges["small (< 50k)"] += 1
            elif amount < 200000:
                ranges["medium (50k-200k)"] += 1
            elif amount < 500000:
                ranges["large (200k-500k)"] += 1
            else:
                ranges["very_large (> 500k)"] += 1
        
        return ranges
    
    def _analyze_temporal_patterns(self, donations: List[Dict]) -> Dict[str, Any]:
        """Analyze when users tend to donate (if timestamp data available)."""
        # For now, return basic analysis since we don't have timestamp data
        # This can be enhanced when temporal data is available
        return {
            "donation_frequency": len(donations),
            "estimated_frequency": "regular" if len(donations) > 5 else "occasional",
            "note": "Temporal analysis requires timestamp data in donations"
        }
    
    def _analyze_project_characteristics(self, enriched_donations: List[Dict]) -> Dict[str, Any]:
        """Analyze characteristics of projects user tends to support."""
        countries = Counter()
        statuses = Counter()
        target_ranges = defaultdict(int)
        
        for donation in enriched_donations:
            project = donation.get('project', {})
            
            # Country analysis
            country = project.get('country', 'Unknown')
            countries[country] += 1
            
            # Status analysis
            status = project.get('is_active', 'Unknown')
            statuses[str(status)] += 1
            
            # Target amount analysis
            target = project.get('target_amount', 0)
            if target < 10000000:  # < 10M
                target_ranges["small_target"] += 1
            elif target < 50000000:  # < 50M
                target_ranges["medium_target"] += 1
            else:
                target_ranges["large_target"] += 1
        
        return {
            "preferred_countries": dict(countries.most_common()),
            "project_statuses": dict(statuses),
            "target_amount_preferences": dict(target_ranges)
        }
    
    def _calculate_generosity_score(self, donations: List[Dict]) -> float:
        """Calculate a generosity score based on donation patterns."""
        if not donations:
            return 0.0
        
        amounts = [d['amount'] for d in donations]
        frequency_score = min(len(donations) / 10, 1.0)  # Max score at 10+ donations
        amount_score = min(sum(amounts) / 1000000, 1.0)  # Max score at 1M+ total
        consistency_score = len(donations) / (len(donations) + 1)  # Reward consistency
        
        return (frequency_score * 0.4 + amount_score * 0.4 + consistency_score * 0.2)
    
    def _calculate_diversity_score(self, enriched_donations: List[Dict]) -> float:
        """Calculate how diverse user's donation portfolio is."""
        if not enriched_donations:
            return 0.0
        
        categories = set()
        countries = set()
        
        for donation in enriched_donations:
            project = donation.get('project', {})
            categories.add(project.get('category_id'))
            countries.add(project.get('country'))
        
        # Remove None values
        categories.discard(None)
        countries.discard(None)
        
        category_diversity = len(categories) / max(len(enriched_donations), 1)
        country_diversity = len(countries) / max(len(enriched_donations), 1)
        
        return (category_diversity + country_diversity) / 2
    
    def find_similar_users(self, target_user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find users with similar donation patterns."""
        target_profile = self.get_user_donation_profile(target_user_id)
        
        if "error" in target_profile:
            return []
        
        similarities = []
        target_categories = set(target_profile["category_preferences"]["preferences"].keys())
        
        for user_id in self.donations.keys():
            if user_id == target_user_id:
                continue
            
            user_profile = self.get_user_donation_profile(user_id)
            if "error" in user_profile:
                continue
            
            # Calculate similarity based on category overlap and other factors
            user_categories = set(user_profile["category_preferences"]["preferences"].keys())
            category_overlap = len(target_categories.intersection(user_categories)) / len(target_categories.union(user_categories)) if target_categories.union(user_categories) else 0
            
            # Amount similarity
            target_avg = target_profile["basic_stats"]["average_amount"]
            user_avg = user_profile["basic_stats"]["average_amount"]
            amount_similarity = 1 - abs(target_avg - user_avg) / max(target_avg, user_avg)
            
            # Generosity similarity
            generosity_similarity = 1 - abs(target_profile["generosity_score"] - user_profile["generosity_score"])
            
            overall_similarity = (category_overlap * 0.5 + amount_similarity * 0.3 + generosity_similarity * 0.2)
            
            similarities.append({
                "user_id": user_id,
                "similarity_score": overall_similarity,
                "category_overlap": category_overlap,
                "amount_similarity": amount_similarity,
                "generosity_similarity": generosity_similarity
            })
        
        return sorted(similarities, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
    
    def get_undonated_recommendations(self, user_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get sophisticated recommendations for undonated projects based on user profile.
        """
        user_profile = self.get_user_donation_profile(user_id)
        
        if "error" in user_profile:
            return []
        
        # Get projects user has already donated to
        user_donations = self.donations.get(user_id, [])
        donated_project_ids = set(str(d['campaign_id']) for d in user_donations)
        
        # Get undonated projects
        undonated_projects = [p for p in self.projects if str(p['id']) not in donated_project_ids]
        
        # Score each undonated project
        scored_projects = []
        for project in undonated_projects:
            score = self._score_project_for_user(project, user_profile)
            scored_projects.append({
                "project": project,
                "recommendation_score": score,
                "reasons": self._get_recommendation_reasons(project, user_profile)
            })
        
        # Sort by score and return top k
        return sorted(scored_projects, key=lambda x: x["recommendation_score"], reverse=True)[:top_k]
    
    def _score_project_for_user(self, project: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Score a project based on how well it matches user's profile."""
        score = 0.0
        
        # Category preference score
        project_category = f"{project.get('category_name', 'Unknown')} (ID: {project.get('category_id')})"
        category_prefs = user_profile["category_preferences"]["preferences"]
        
        if project_category in category_prefs:
            score += category_prefs[project_category]["combined_score"] * 0.4
        
        # Country preference score
        project_country = project.get('country', 'Unknown')
        country_prefs = user_profile["project_characteristics"]["preferred_countries"]
        
        if project_country in country_prefs:
            country_weight = country_prefs[project_country] / user_profile["basic_stats"]["total_donations"]
            score += country_weight * 0.2
        
        # Target amount compatibility
        project_target = project.get('target_amount', 0)
        user_avg_amount = user_profile["basic_stats"]["average_amount"]
        
        # Prefer projects where user's typical donation is meaningful (0.1% to 10% of target)
        if project_target > 0:
            donation_impact = user_avg_amount / project_target
            if 0.001 <= donation_impact <= 0.1:  # Sweet spot for impact
                score += 0.2
            elif donation_impact > 0.1:  # Very high impact
                score += 0.3
        
        # Active project bonus
        if project.get('is_active', False):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_recommendation_reasons(self, project: Dict[str, Any], user_profile: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasons for recommendation."""
        reasons = []
        
        # Category match
        project_category = f"{project.get('category_name', 'Unknown')} (ID: {project.get('category_id')})"
        category_prefs = user_profile["category_preferences"]["preferences"]
        
        if project_category in category_prefs:
            pref_score = category_prefs[project_category]["combined_score"]
            if pref_score > 0.3:
                reasons.append(f"Matches your strong preference for {project.get('category_name')} projects")
            else:
                reasons.append(f"Aligns with your interest in {project.get('category_name')} causes")
        
        # Country match
        project_country = project.get('country', 'Unknown')
        country_prefs = user_profile["project_characteristics"]["preferred_countries"]
        
        if project_country in country_prefs and country_prefs[project_country] > 1:
            reasons.append(f"Located in {project_country}, where you've supported projects before")
        
        # Impact potential
        project_target = project.get('target_amount', 0)
        user_avg_amount = user_profile["basic_stats"]["average_amount"]
        
        if project_target > 0:
            donation_impact = user_avg_amount / project_target
            if donation_impact > 0.01:
                reasons.append("Your typical donation would have significant impact on this project")
        
        # Active status
        if project.get('is_active', False):
            reasons.append("Project is currently active and accepting donations")
        
        if not reasons:
            reasons.append("Recommended based on your overall donation patterns")
        
        return reasons


def example_usage():
    """Demonstrate the enhanced donation analysis capabilities."""
    analyzer = DonationAnalyzer()
    
    print("=== ENHANCED DONATION ANALYSIS DEMO ===\n")
    
    # Analyze user 1001
    user_id = "1001"
    print(f"📊 Comprehensive Profile for User {user_id}:")
    print("=" * 50)
    
    profile = analyzer.get_user_donation_profile(user_id)
    
    # Basic stats
    basic = profile["basic_stats"]
    print(f"💰 Total Donations: {basic['total_donations']}")
    print(f"💰 Total Amount: IDR {basic['total_amount']:,}")
    print(f"💰 Average Amount: IDR {basic['average_amount']:,.0f}")
    print(f"💰 Unique Campaigns: {basic['unique_projects']}")
    print(f"💰 Generosity Score: {profile['generosity_score']:.2f}/1.0")
    print(f"💰 Diversity Score: {profile['diversity_score']:.2f}/1.0")
    
    # Category preferences
    print(f"\n🎯 Top Category Preferences:")
    for i, (category, stats) in enumerate(list(profile["category_preferences"]["preferences"].items())[:3], 1):
        print(f"  {i}. {category}")
        print(f"     - Combined Score: {stats['combined_score']:.2f}")
        print(f"     - Donations: {stats['donation_count']} ({stats['frequency_weight']:.1%})")
        print(f"     - Amount: IDR {stats['total_amount']:,} ({stats['amount_weight']:.1%})")
    
    # Find similar users
    print(f"\n👥 Users with Similar Donation Patterns:")
    similar_users = analyzer.find_similar_users(user_id, top_k=3)
    for i, user in enumerate(similar_users, 1):
        print(f"  {i}. User {user['user_id']} (Similarity: {user['similarity_score']:.2f})")
        print(f"     - Category Overlap: {user['category_overlap']:.2f}")
        print(f"     - Amount Similarity: {user['amount_similarity']:.2f}")
    
    # Get sophisticated recommendations
    print(f"\n🎯 Smart Recommendations for Undonated Projects:")
    recommendations = analyzer.get_undonated_recommendations(user_id, top_k=5)
    
    for i, rec in enumerate(recommendations, 1):
        project = rec["project"]
        print(f"\n  {i}. {project['title']}")
        print(f"     Category: {project.get('category_name', 'Unknown')}")
        print(f"     Target: IDR {project.get('target_amount', 0):,}")
        print(f"     Country: {project.get('country', 'Unknown')}")
        print(f"     Recommendation Score: {rec['recommendation_score']:.2f}")
        print(f"     Reasons:")
        for reason in rec["reasons"]:
            print(f"       • {reason}")


if __name__ == "__main__":
    example_usage()