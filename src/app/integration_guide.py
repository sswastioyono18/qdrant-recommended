"""
COMPREHENSIVE RECOMMENDATION SYSTEM INTEGRATION GUIDE

This module demonstrates how to integrate and use all the enhanced recommendation
system components to leverage donation history for recommending undonated campaigns.

Components included:
1. DonationAnalyzer - Enhanced donation pattern analysis
2. AdvancedUserProfiler - Comprehensive user profiling
3. SmartCampaignFilter - Intelligent campaign filtering
4. ComprehensiveRecommendationEngine - Multi-approach recommendations
5. RecommendationEvaluator - System evaluation and testing

This guide shows practical integration patterns and usage examples.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from donation_analyzer import DonationAnalyzer
from advanced_profiler import AdvancedUserProfiler
from smart_filter import SmartCampaignFilter
from comprehensive_recommender import ComprehensiveRecommendationEngine
from recommendation_evaluator import RecommendationEvaluator


class EnhancedRecommendationSystem:
    """
    Main integration class that orchestrates all recommendation components.
    This is the primary interface for getting enhanced recommendations.
    """
    
    def __init__(self, donations_path: str = "../../data/donations.json", 
                 campaigns_path: str = "../../data/campaigns.json"):
        """Initialize the enhanced recommendation system."""
        self.donations_path = donations_path
        self.campaigns_path = campaigns_path
        
        # Initialize all components
        print("🚀 Initializing Enhanced Recommendation System...")
        
        self.analyzer = DonationAnalyzer(donations_path, campaigns_path)
        self.profiler = AdvancedUserProfiler(donations_path, campaigns_path)
        self.smart_filter = SmartCampaignFilter(donations_path, campaigns_path)
        self.comprehensive_engine = ComprehensiveRecommendationEngine(donations_path, campaigns_path)
        self.evaluator = RecommendationEvaluator(donations_path, campaigns_path)
        
        print("✅ Enhanced Recommendation System initialized successfully!")
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive insights about a user's donation behavior.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing user insights and profile information
        """
        print(f"🔍 Analyzing user {user_id}...")
        
        # Get donation analysis
        donation_profile = self.analyzer.get_user_donation_profile(user_id)
        
        # Get advanced user profile
        advanced_profile = self.profiler.create_comprehensive_profile(user_id)
        
        # Get filtering strategy
        filter_summary = self.smart_filter._get_filtering_strategy(advanced_profile)
        
        return {
            "user_id": user_id,
            "donation_analysis": donation_profile,
            "advanced_profile": advanced_profile,
            "filtering_strategy": filter_summary,
            "insights_generated_at": datetime.now().isoformat()
        }
    
    def get_enhanced_recommendations(self, user_id: str, max_recommendations: int = 10,
                                   include_explanations: bool = True,
                                   include_insights: bool = False) -> Dict[str, Any]:
        """
        Get enhanced recommendations using the comprehensive system.
        
        Args:
            user_id: User identifier
            max_recommendations: Maximum number of recommendations
            include_explanations: Include detailed explanations
            include_insights: Include user insights in response
            
        Returns:
            Enhanced recommendations with optional insights
        """
        print(f"🎯 Generating enhanced recommendations for user {user_id}...")
        
        # Get comprehensive recommendations
        recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
            user_id, max_recommendations, include_explanations
        )
        
        # Add user insights if requested
        if include_insights:
            recommendations["user_insights"] = self.get_user_insights(user_id)
        
        return recommendations
    
    def compare_users(self, user_id_1: str, user_id_2: str) -> Dict[str, Any]:
        """
        Compare two users' donation patterns and recommendation strategies.
        
        Args:
            user_id_1: First user identifier
            user_id_2: Second user identifier
            
        Returns:
            Comparison analysis between the two users
        """
        print(f"⚖️ Comparing users {user_id_1} and {user_id_2}...")
        
        # Get insights for both users
        user1_insights = self.get_user_insights(user_id_1)
        user2_insights = self.get_user_insights(user_id_2)
        
        # Get recommendations for both users
        user1_recs = self.get_enhanced_recommendations(user_id_1, 5, False)
        user2_recs = self.get_enhanced_recommendations(user_id_2, 5, False)
        
        # Compare donation patterns
        user1_donations = user1_insights["donation_analysis"]
        user2_donations = user2_insights["donation_analysis"]
        
        comparison = {
            "user_comparison": {
                "user_1": {
                    "id": user_id_1,
                    "total_donations": user1_donations.get("total_donations", 0),
                    "total_amount": user1_donations.get("total_amount", 0),
                    "behavioral_type": user1_insights["advanced_profile"].get("behavioral_profile", {}).get("behavioral_type", "unknown"),
                    "recommendation_confidence": user1_insights["advanced_profile"].get("recommendation_confidence", 0),
                    "top_categories": list(user1_donations.get("category_analysis", {}).get("category_preferences", {}).keys())[:3]
                },
                "user_2": {
                    "id": user_id_2,
                    "total_donations": user2_donations.get("total_donations", 0),
                    "total_amount": user2_donations.get("total_amount", 0),
                    "behavioral_type": user2_insights["advanced_profile"].get("behavioral_profile", {}).get("behavioral_type", "unknown"),
                    "recommendation_confidence": user2_insights["advanced_profile"].get("recommendation_confidence", 0),
                    "top_categories": list(user2_donations.get("category_analysis", {}).get("category_preferences", {}).keys())[:3]
                }
            },
            "recommendation_comparison": {
                "user_1_recommendations": len(user1_recs.get("recommendations", [])),
                "user_2_recommendations": len(user2_recs.get("recommendations", [])),
                "common_campaigns": self._find_common_recommended_campaigns(user1_recs, user2_recs),
                "strategy_differences": {
                    "user_1_strategy": user1_recs.get("recommendation_strategy", "unknown"),
                    "user_2_strategy": user2_recs.get("recommendation_strategy", "unknown")
                }
            },
            "similarity_analysis": {
                "donation_pattern_similarity": self._calculate_pattern_similarity(user1_donations, user2_donations),
                "behavioral_similarity": user1_insights["advanced_profile"].get("behavioral_profile", {}).get("behavioral_type") == 
                                       user2_insights["advanced_profile"].get("behavioral_profile", {}).get("behavioral_type")
            }
        }
        
        return comparison
    
    def evaluate_system_performance(self, test_users: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the performance of the enhanced recommendation system.
        
        Args:
            test_users: List of user IDs to test (if None, uses default set)
            
        Returns:
            Comprehensive evaluation results
        """
        print("📊 Evaluating system performance...")
        
        if test_users is None:
            # Use a default set of test users
            test_users = ["1001", "2002", "3003", "4004", "5005"]
        
        return self.evaluator.run_comprehensive_evaluation(test_users, 5)
    
    def get_campaign_analytics(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific campaign based on user interactions.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            Campaign analytics and recommendation insights
        """
        print(f"📈 Analyzing campaign {campaign_id}...")
        
        # Find users who donated to this campaign
        donors = []
        for user_id, donations in self.analyzer.donations.items():
            for donation in donations:
                if str(donation.get('campaign_id')) == str(campaign_id):
                    donors.append(user_id)
        
        # Find campaign details
        campaign = None
        for c in self.analyzer.campaigns:
            if str(c['id']) == str(campaign_id):
                campaign = c
                break
        
        if not campaign:
            return {"error": f"Campaign {campaign_id} not found"}
        
        # Analyze donor profiles
        donor_profiles = []
        for donor_id in donors:
            profile = self.profiler.create_comprehensive_profile(donor_id)
            if "error" not in profile:
                donor_profiles.append({
                    "user_id": donor_id,
                    "behavioral_type": profile.get("behavioral_profile", {}).get("behavioral_type", "unknown"),
                    "generosity_score": profile.get("behavioral_profile", {}).get("generosity_score", 0),
                    "category_preferences": list(profile.get("category_preferences", {}).get("preferences", {}).keys())[:3]
                })
        
        # Calculate recommendation frequency
        recommendation_count = 0
        total_users_checked = 0
        
        # Sample a few users to see how often this campaign is recommended
        sample_users = list(self.analyzer.donations.keys())[:10]
        for user_id in sample_users:
            total_users_checked += 1
            recs = self.get_enhanced_recommendations(user_id, 10, False)
            if "recommendations" in recs:
                for rec in recs["recommendations"]:
                    if str(rec.get("campaign_id")) == str(campaign_id):
                        recommendation_count += 1
                        break
        
        return {
            "campaign_id": campaign_id,
            "campaign_details": campaign,
            "donor_analysis": {
                "total_donors": len(donors),
                "donor_profiles": donor_profiles,
                "common_behavioral_types": self._get_common_behavioral_types(donor_profiles),
                "average_generosity_score": sum(p.get("generosity_score", 0) for p in donor_profiles) / len(donor_profiles) if donor_profiles else 0
            },
            "recommendation_analytics": {
                "recommendation_frequency": recommendation_count / total_users_checked if total_users_checked > 0 else 0,
                "sample_size": total_users_checked,
                "estimated_appeal": "high" if recommendation_count / total_users_checked > 0.3 else "medium" if recommendation_count / total_users_checked > 0.1 else "low"
            }
        }
    
    def _find_common_recommended_campaigns(self, recs1: Dict, recs2: Dict) -> List[str]:
        """Find campaigns recommended to both users."""
        campaigns1 = set(str(rec.get("campaign_id")) for rec in recs1.get("recommendations", []))
        campaigns2 = set(str(rec.get("campaign_id")) for rec in recs2.get("recommendations", []))
        return list(campaigns1.intersection(campaigns2))
    
    def _calculate_pattern_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Calculate similarity between two donation patterns."""
        # Simple similarity based on category overlap
        cats1 = set(profile1.get("category_analysis", {}).get("category_preferences", {}).keys())
        cats2 = set(profile2.get("category_analysis", {}).get("category_preferences", {}).keys())
        
        if not cats1 and not cats2:
            return 1.0
        if not cats1 or not cats2:
            return 0.0
        
        intersection = len(cats1.intersection(cats2))
        union = len(cats1.union(cats2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_common_behavioral_types(self, profiles: List[Dict]) -> Dict[str, int]:
        """Get frequency of behavioral types among profiles."""
        types = {}
        for profile in profiles:
            behavioral_type = profile.get("behavioral_type", "unknown")
            types[behavioral_type] = types.get(behavioral_type, 0) + 1
        return types


def demonstrate_integration():
    """Demonstrate the complete integration of the enhanced recommendation system."""
    print("🌟 ENHANCED RECOMMENDATION SYSTEM INTEGRATION DEMO")
    print("=" * 80)
    
    # Initialize the system
    system = EnhancedRecommendationSystem()
    
    print("\n1️⃣ USER INSIGHTS ANALYSIS")
    print("-" * 40)
    
    # Get comprehensive insights for a user
    user_insights = system.get_user_insights("1001")
    print(f"User 1001 Profile:")
    print(f"  • Total Donations: {user_insights['donation_analysis'].get('total_donations', 0)}")
    print(f"  • Behavioral Type: {user_insights['advanced_profile'].get('behavioral_profile', {}).get('behavioral_type', 'unknown')}")
    print(f"  • Recommendation Confidence: {user_insights['advanced_profile'].get('recommendation_confidence', 0):.2f}")
    print(f"  • Filtering Strategy: {user_insights['filtering_strategy'].get('strategy_name', 'unknown')}")
    
    print("\n2️⃣ ENHANCED RECOMMENDATIONS")
    print("-" * 40)
    
    # Get enhanced recommendations
    recommendations = system.get_enhanced_recommendations("1001", 3, True, False)
    print(f"Generated {len(recommendations.get('recommendations', []))} recommendations for user 1001:")
    
    for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
        print(f"  {i}. {rec.get('title', 'Unknown')} (Score: {rec.get('final_score', 0):.3f})")
        if 'explanation' in rec:
            print(f"     Reason: {rec['explanation'][:100]}...")
    
    print("\n3️⃣ USER COMPARISON")
    print("-" * 40)
    
    # Compare two users
    comparison = system.compare_users("1001", "2002")
    user1_data = comparison['user_comparison']['user_1']
    user2_data = comparison['user_comparison']['user_2']
    
    print(f"User Comparison:")
    print(f"  • User 1001: {user1_data['total_donations']} donations, {user1_data['behavioral_type']} type")
    print(f"  • User 2002: {user2_data['total_donations']} donations, {user2_data['behavioral_type']} type")
    print(f"  • Pattern Similarity: {comparison['similarity_analysis']['donation_pattern_similarity']:.3f}")
    print(f"  • Common Recommended Campaigns: {len(comparison['recommendation_comparison']['common_campaigns'])}")
    
    print("\n4️⃣ CAMPAIGN ANALYTICS")
    print("-" * 40)
    
    # Analyze a specific campaign
    campaign_analytics = system.get_campaign_analytics("18000808")
    if "error" not in campaign_analytics:
        donor_analysis = campaign_analytics['donor_analysis']
        rec_analytics = campaign_analytics['recommendation_analytics']
        
        print(f"Campaign 18000808 Analytics:")
        print(f"  • Total Donors: {donor_analysis['total_donors']}")
        print(f"  • Average Generosity Score: {donor_analysis['average_generosity_score']:.3f}")
        print(f"  • Recommendation Frequency: {rec_analytics['recommendation_frequency']:.3f}")
        print(f"  • Estimated Appeal: {rec_analytics['estimated_appeal']}")
    
    print("\n5️⃣ SYSTEM PERFORMANCE EVALUATION")
    print("-" * 40)
    
    # Evaluate system performance
    evaluation = system.evaluate_system_performance(["1001", "2002", "3003"])
    
    accuracy = evaluation.get('accuracy_metrics', {})
    performance = evaluation.get('performance_metrics', {})
    
    print(f"System Performance:")
    print(f"  • Mean Relevance Score: {accuracy.get('recommendation_relevance', {}).get('mean_score', 0):.3f}")
    print(f"  • Success Rate: {performance.get('success_rate', {}).get('success_percentage', 0):.1f}%")
    print(f"  • Mean Response Time: {performance.get('response_time', {}).get('mean_seconds', 0):.3f}s")
    
    print("\n✅ INTEGRATION DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    return {
        "user_insights": user_insights,
        "recommendations": recommendations,
        "user_comparison": comparison,
        "campaign_analytics": campaign_analytics,
        "system_evaluation": evaluation
    }


def integration_best_practices():
    """Display best practices for integrating the enhanced recommendation system."""
    print("\n📋 INTEGRATION BEST PRACTICES")
    print("=" * 80)
    
    practices = [
        "1. Initialize the EnhancedRecommendationSystem once and reuse the instance",
        "2. Use get_user_insights() for comprehensive user analysis before recommendations",
        "3. Set appropriate max_recommendations based on your UI constraints (5-10 typical)",
        "4. Include explanations for user-facing recommendations to build trust",
        "5. Use compare_users() to understand user segments and personalization opportunities",
        "6. Regularly run evaluate_system_performance() to monitor recommendation quality",
        "7. Use campaign_analytics() to understand which campaigns appeal to which user types",
        "8. Cache user profiles when possible to improve response times",
        "9. Monitor recommendation diversity to avoid filter bubbles",
        "10. A/B test different recommendation strategies using the evaluation framework"
    ]
    
    for practice in practices:
        print(f"  {practice}")
    
    print("\n🔧 TECHNICAL INTEGRATION NOTES")
    print("-" * 40)
    
    notes = [
        "• All components are designed to work with the existing data format",
        "• The system automatically filters out campaigns users have already donated to",
        "• Recommendation scores are normalized between 0 and 1 for consistency",
        "• The system gracefully handles missing data and provides fallback strategies",
        "• All components include error handling and return structured error messages",
        "• The evaluation framework provides comprehensive metrics for system monitoring"
    ]
    
    for note in notes:
        print(f"  {note}")


if __name__ == "__main__":
    # Run the integration demonstration
    demo_results = demonstrate_integration()
    
    # Show best practices
    integration_best_practices()
    
    print(f"\n💾 Demo completed successfully!")
    print(f"🎯 The enhanced recommendation system is ready for production integration!")