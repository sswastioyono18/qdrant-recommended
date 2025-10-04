"""
COMPREHENSIVE RECOMMENDATION SYSTEM INTEGRATION GUIDE - Database Version

This module demonstrates how to integrate and use all the enhanced recommendation
system components with database integration to leverage donation history for 
recommending undonated campaigns.

Components included:
1. DonationAnalyzerDB - Enhanced donation pattern analysis with database
2. AdvancedUserProfilerDB - Comprehensive user profiling with database
3. SmartCampaignFilterDB - Intelligent campaign filtering with database
4. ComprehensiveRecommendationEngineDB - Multi-approach recommendations with database
5. Database models and repositories for data access

This guide shows practical integration patterns and usage examples for database-enabled system.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import os

from .donation_analyzer_db import DatabaseDonationAnalyzer
from .advanced_profiler_db import AdvancedUserProfilerDB
from .smart_filter_db import SmartProjectFilterDB
from .comprehensive_recommender_db import ComprehensiveRecommendationEngineDB
from .database import DatabaseConfig, DatabaseManager
from .models import ProjectRepository, DonationRepository, UserRepository, Project, Donation, User


class EnhancedRecommendationSystemDB:
    """
    Main integration class that orchestrates all recommendation components with database.
    This is the primary interface for getting enhanced recommendations from database.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the enhanced recommendation system with database manager."""
        self.db_manager = db_manager
        
        # Initialize all components
        print("🚀 Initializing Enhanced Recommendation System with Database...")
        
        self.analyzer = DatabaseDonationAnalyzer(auto_init_db=True)
        self.profiler = AdvancedUserProfilerDB(db_manager)
        self.smart_filter = SmartProjectFilterDB(db_manager)
        self.comprehensive_engine = ComprehensiveRecommendationEngineDB(db_manager)
        
        # Initialize repositories
        self.project_repo = ProjectRepository(db_manager)
        self.donation_repo = DonationRepository(db_manager)
        self.user_repo = UserRepository(db_manager)
        
        print("✅ All components initialized successfully!")
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive insights about a user's donation behavior and preferences.
        
        Args:
            user_id: User ID to analyze
        
        Returns:
            Dictionary containing user insights and profile information
        """
        print(f"🔍 Analyzing user {user_id}...")
        
        # Get comprehensive user profile
        user_profile = self.profiler.create_comprehensive_profile(user_id)
        
        if "error" in user_profile:
            return {"error": f"Could not analyze user {user_id}"}
        
        # Get donation history analysis
        print('calling get_user_donation_profile from get_user_insights')
        donation_profile = self.analyzer.get_user_donation_profile(user_id)

        # Get smart filtering insights
        print('calling get_smart_filtered_projects from get_user_insights')
        smart_filter_result = self.smart_filter.get_smart_filtered_projects(
            user_id=user_id, max_projects=5, include_reasoning=False
        )
        
        return {
            "user_id": user_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "behavioral_profile": user_profile.get("behavioral_profile", {}),
            "preference_profile": user_profile.get("preference_profile", {}),
            "social_profile": user_profile.get("social_profile", {}),
            "predictive_profile": user_profile.get("predictive_profile", {}),
            "donation_statistics": donation_profile.get("basic_stats", {}),
            "filtering_strategy": smart_filter_result.get("filtering_strategy", {}),
            "total_donations": user_profile.get("total_donations", 0),
            "insights_summary": self._create_insights_summary(user_profile)
        }
    
    def get_enhanced_recommendations(self, user_id: str, max_recommendations: int = 10,
                                   include_explanations: bool = True,
                                   include_insights: bool = False) -> Dict[str, Any]:
        """
        Get enhanced recommendations using the comprehensive engine.
        
        Args:
            user_id: User ID to get recommendations for
            max_recommendations: Maximum number of recommendations
            include_explanations: Whether to include detailed explanations
            include_insights: Whether to include user insights
        
        Returns:
            Dictionary containing recommendations and optional insights
        """
        print(f"🎯 Generating enhanced recommendations for user {user_id}...")
        
        # Get comprehensive recommendations
        recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
            user_id=user_id,
            max_recommendations=max_recommendations,
            include_explanations=include_explanations,
            diversity_factor=0.3
        )
        
        result = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        }
        
        # Add user insights if requested
        if include_insights:
            result["user_insights"] = self.get_user_insights(user_id)
        
        return result
    
    def compare_users(self, user_id_1: str, user_id_2: str) -> Dict[str, Any]:
        """
        Compare two users' profiles and recommendation patterns.
        
        Args:
            user_id_1: First user ID
            user_id_2: Second user ID
        
        Returns:
            Dictionary containing comparison results
        """
        print(f"🔄 Comparing users {user_id_1} and {user_id_2}...")
        
        # Get profiles for both users
        profile1 = self.profiler.create_comprehensive_profile(user_id_1)
        profile2 = self.profiler.create_comprehensive_profile(user_id_2)
        
        if "error" in profile1 or "error" in profile2:
            return {"error": "Could not create profiles for comparison"}
        
        # Get recommendations for both users
        recs1 = self.comprehensive_engine.get_comprehensive_recommendations(
            user_id_1, max_recommendations=10, include_explanations=False
        )
        recs2 = self.comprehensive_engine.get_comprehensive_recommendations(
            user_id_2, max_recommendations=10, include_explanations=False
        )
        
        # Find common recommended campaigns
        common_campaigns = self._find_common_recommended_campaigns(recs1, recs2)
        
        # Calculate similarity metrics
        behavioral_similarity = self._calculate_behavioral_similarity(profile1, profile2)
        preference_similarity = self._calculate_preference_similarity(profile1, profile2)
        
        return {
            "user_comparison": {
                "user_1": {
                    "user_id": user_id_1,
                    "behavioral_type": profile1.get("behavioral_profile", {}).get("behavioral_type", "unknown"),
                    "engagement_level": profile1.get("behavioral_profile", {}).get("engagement_level", {}).get("level", "unknown"),
                    "total_donations": profile1.get("total_donations", 0),
                    "avg_amount": profile1.get("avg_amount", 0)
                },
                "user_2": {
                    "user_id": user_id_2,
                    "behavioral_type": profile2.get("behavioral_profile", {}).get("behavioral_type", "unknown"),
                    "engagement_level": profile2.get("behavioral_profile", {}).get("engagement_level", {}).get("level", "unknown"),
                    "total_donations": profile2.get("total_donations", 0),
                    "avg_amount": profile2.get("avg_amount", 0)
                }
            },
            "similarity_metrics": {
                "behavioral_similarity": behavioral_similarity,
                "preference_similarity": preference_similarity,
                "overall_similarity": (behavioral_similarity + preference_similarity) / 2
            },
            "recommendation_overlap": {
                "common_campaigns": common_campaigns,
                "overlap_count": len(common_campaigns),
                "user_1_total_recs": len(recs1.get("recommendations", [])),
                "user_2_total_recs": len(recs2.get("recommendations", [])),
                "overlap_percentage": len(common_campaigns) / max(len(recs1.get("recommendations", [])), 1) * 100
            }
        }
    
    def evaluate_system_performance(self, test_users: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the performance of the recommendation system.
        
        Args:
            test_users: List of user IDs to test (if None, uses sample from database)
        
        Returns:
            Dictionary containing performance metrics
        """
        print("📊 Evaluating system performance...")
        
        if test_users is None:
            # Get sample users from database
            all_users = self.user_repo.get_all()
            test_users = [str(user.id) for user in all_users[:10]]  # Test with first 10 users
        
        performance_results = []
        
        for user_id in test_users:
            try:
                result = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations=5, include_explanations=False
                )
                
                if "error" not in result:
                    performance_results.append({
                        "user_id": user_id,
                        "recommendations_count": len(result.get("recommendations", [])),
                        "avg_score": sum(r.get("final_score", 0) for r in result.get("recommendations", [])) / max(len(result.get("recommendations", [])), 1),
                        "performance_metrics": result.get("performance_metrics", {})
                    })
            except Exception as e:
                print(f"❌ Error evaluating user {user_id}: {str(e)}")
        
        # Calculate overall metrics
        if performance_results:
            avg_recommendations = sum(r["recommendations_count"] for r in performance_results) / len(performance_results)
            avg_score = sum(r["avg_score"] for r in performance_results) / len(performance_results)
        else:
            avg_recommendations = 0
            avg_score = 0
        
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "users_tested": len(test_users),
            "successful_evaluations": len(performance_results),
            "success_rate": (len(performance_results) / len(test_users) * 100) if test_users else 0,
            "average_recommendations_per_user": avg_recommendations,
            "average_recommendation_score": avg_score,
            "detailed_results": performance_results
        }
    
    def get_campaign_analytics(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific campaign.
        
        Args:
            campaign_id: Campaign ID to analyze
        
        Returns:
            Dictionary containing campaign analytics
        """
        print(f"📈 Analyzing campaign {campaign_id}...")
        
        # Get project details
        project = self.project_repo.get_by_id(campaign_id)
        if not project:
            return {"error": f"Project {campaign_id} not found"}
        
        # Get all donations for this campaign
        campaign_donations = self.donation_repo.get_by_campaign_id(campaign_id)
        
        # Calculate analytics
        total_raised = sum(donation.amount for donation in campaign_donations)
        donor_count = len(set(donation.user_id for donation in campaign_donations))
        avg_donation = total_raised / len(campaign_donations) if campaign_donations else 0
        
        # Get donor profiles
        donor_profiles = []
        for donation in campaign_donations[:10]:  # Analyze first 10 donors
            try:
                profile = self.profiler.create_comprehensive_profile(donation.user_id)
                if "error" not in profile:
                    donor_profiles.append({
                        "user_id": donation.user_id,
                        "behavioral_type": profile.get("behavioral_profile", {}).get("behavioral_type", "unknown"),
                        "donation_amount": donation.amount
                    })
            except Exception:
                continue
        
        # Analyze donor behavioral types
        behavioral_types = {}
        for profile in donor_profiles:
            btype = profile["behavioral_type"]
            behavioral_types[btype] = behavioral_types.get(btype, 0) + 1
        
        return {
            "campaign_id": campaign_id,
            "campaign_details": project.to_dict(),
            "funding_analytics": {
                "total_raised": total_raised,
                "target_amount": project.target_amount,
                "funding_percentage": (total_raised / project.target_amount * 100) if project.target_amount > 0 else 0,
                "donor_count": donor_count,
                "total_donations": len(campaign_donations),
                "average_donation": avg_donation
            },
            "donor_analytics": {
                "behavioral_type_distribution": behavioral_types,
                "sample_donor_profiles": donor_profiles
            },
            "recommendation_potential": {
                "estimated_interested_users": self._estimate_interested_users(project),
                "recommendation_score": self._calculate_campaign_recommendation_score(project, campaign_donations)
            }
        }
    
    def _find_common_recommended_campaigns(self, recs1: Dict, recs2: Dict) -> List[str]:
        """Find campaigns that appear in both recommendation lists."""
        campaigns1 = {str(r.get("id", "")) for r in recs1.get("recommendations", [])}
        campaigns2 = {str(r.get("id", "")) for r in recs2.get("recommendations", [])}
        return list(campaigns1.intersection(campaigns2))
    
    def _calculate_behavioral_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Calculate behavioral similarity between two user profiles."""
        behavioral1 = profile1.get("behavioral_profile", {})
        behavioral2 = profile2.get("behavioral_profile", {})
        
        # Compare behavioral types
        type_match = 1.0 if behavioral1.get("behavioral_type") == behavioral2.get("behavioral_type") else 0.0
        
        # Compare engagement levels
        engagement1 = behavioral1.get("engagement_level", {}).get("level", "")
        engagement2 = behavioral2.get("engagement_level", {}).get("level", "")
        engagement_match = 1.0 if engagement1 == engagement2 else 0.5 if engagement1 and engagement2 else 0.0
        
        # Compare risk tolerance
        risk1 = behavioral1.get("risk_tolerance", {}).get("tolerance", "")
        risk2 = behavioral2.get("risk_tolerance", {}).get("tolerance", "")
        risk_match = 1.0 if risk1 == risk2 else 0.5 if risk1 and risk2 else 0.0
        
        return (type_match + engagement_match + risk_match) / 3
    
    def _calculate_preference_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Calculate preference similarity between two user profiles."""
        pref1 = profile1.get("preference_profile", {})
        pref2 = profile2.get("preference_profile", {})
        
        # Compare category preferences
        cat_prefs1 = set(pref1.get("category_semantics", {}).get("primary_categories", []))
        cat_prefs2 = set(pref2.get("category_semantics", {}).get("primary_categories", []))
        
        if cat_prefs1 and cat_prefs2:
            category_similarity = len(cat_prefs1.intersection(cat_prefs2)) / len(cat_prefs1.union(cat_prefs2))
        else:
            category_similarity = 0.0
        
        # Compare amount preferences (simplified)
        avg1 = profile1.get("avg_amount", 0)
        avg2 = profile2.get("avg_amount", 0)
        
        if avg1 > 0 and avg2 > 0:
            amount_similarity = 1 - abs(avg1 - avg2) / max(avg1, avg2)
        else:
            amount_similarity = 0.5
        
        return (category_similarity + amount_similarity) / 2
    
    def _create_insights_summary(self, user_profile: Dict[str, Any]) -> Dict[str, str]:
        """Create a human-readable summary of user insights."""
        behavioral = user_profile.get("behavioral_profile", {})
        preference = user_profile.get("preference_profile", {})
        
        behavioral_type = behavioral.get("behavioral_type", "unknown")
        engagement_level = behavioral.get("engagement_level", {}).get("level", "unknown")
        primary_categories = preference.get("category_semantics", {}).get("primary_categories", [])
        # Normalize primary categories to a list of strings (handle tuples like (category, score))
        normalized_primary_categories = []
        for cat in primary_categories:
            try:
                name = cat[0] if isinstance(cat, (list, tuple)) else cat
                if name is None:
                    continue
                normalized_primary_categories.append(str(name))
            except Exception:
                # Skip malformed entries
                continue
        
        return {
            "user_type": f"{behavioral_type.replace('_', ' ').title()} with {engagement_level} engagement",
            "donation_pattern": f"Prefers {', '.join(normalized_primary_categories[:3]) if normalized_primary_categories else 'various'} categories",
            "recommendation_strategy": f"Best approached with {behavioral_type.replace('_', ' ')} focused recommendations"
        }
    
    def _estimate_interested_users(self, project: Project) -> int:
        """Estimate how many users might be interested in this project."""
        # Get users who have donated to similar categories
        similar_donations = self.donation_repo.get_all()
        interested_users = set()
        
        for donation in similar_donations:
            donor_project = self.project_repo.get_by_id(donation.project_id)
            # Safely compare categories by normalizing to lowercase strings
            def _norm_category(val: Any) -> str:
                if val is None:
                    return ""
                try:
                    return str(val).lower()
                except Exception:
                    return ""
            if donor_project and _norm_category(getattr(donor_project, 'category', None)) == _norm_category(getattr(project, 'category', None)):
                interested_users.add(donation.user_id)
        
        return len(interested_users)
    
    def _calculate_campaign_recommendation_score(self, campaign: Project, donations: List[Donation]) -> float:
        """Calculate how recommendable a campaign is."""
        # Base score on funding progress and donor engagement
        funding_ratio = sum(d.amount for d in donations) / campaign.target_amount if campaign.target_amount > 0 else 0
        donor_count = len(set(d.user_id for d in donations))
        
        # Normalize scores
        funding_score = min(funding_ratio, 1.0)
        engagement_score = min(donor_count / 50, 1.0)  # Normalize to 50 donors max
        
        return (funding_score + engagement_score) / 2


def demonstrate_integration():
    """Demonstrate the integration of all components with database."""
    print("🎯 ENHANCED RECOMMENDATION SYSTEM - DATABASE INTEGRATION DEMO")
    print("=" * 70)
    
    # Initialize database
    config = DatabaseConfig()
    db_manager = DatabaseManager(config)
    
    # Initialize the enhanced system
    system = EnhancedRecommendationSystemDB(db_manager)
    
    # Demo user ID (you should have this user in your database)
    demo_user_id = "1001"
    
    print(f"\n1️⃣ USER INSIGHTS ANALYSIS")
    print("-" * 30)
    
    insights = system.get_user_insights(demo_user_id)
    if "error" not in insights:
        print(f"✅ User {demo_user_id} Analysis:")
        print(f"   Behavioral Type: {insights.get('behavioral_profile', {}).get('behavioral_type', 'Unknown')}")
        print(f"   Engagement Level: {insights.get('behavioral_profile', {}).get('engagement_level', {}).get('level', 'Unknown')}")
        print(f"   Total Donations: {insights.get('total_donations', 0)}")
        print(f"   Primary Categories: {insights.get('preference_profile', {}).get('category_semantics', {}).get('primary_categories', [])}")
    else:
        print(f"❌ Could not analyze user {demo_user_id}: {insights.get('error')}")
    
    print(f"\n2️⃣ ENHANCED RECOMMENDATIONS")
    print("-" * 30)
    
    recommendations = system.get_enhanced_recommendations(
        demo_user_id, 
        max_recommendations=5, 
        include_explanations=True
    )
    
    if "error" not in recommendations.get("recommendations", {}):
        recs = recommendations["recommendations"]["recommendations"]
        print(f"✅ Generated {len(recs)} recommendations for user {demo_user_id}:")
        
        for i, rec in enumerate(recs[:3], 1):
            print(f"\n   {i}. {rec.get('title', 'Unknown Title')}")
            print(f"      Score: {rec.get('final_score', 0):.3f}")
            print(f"      Category: {rec.get('category', 'Unknown')}")
            print(f"      Target: IDR {rec.get('target_amount', 0):,}")
    else:
        print(f"❌ Could not generate recommendations: {recommendations.get('error')}")
    
    print(f"\n3️⃣ SYSTEM PERFORMANCE EVALUATION")
    print("-" * 30)
    
    performance = system.evaluate_system_performance()
    print(f"✅ System Performance:")
    print(f"   Users Tested: {performance.get('users_tested', 0)}")
    print(f"   Success Rate: {performance.get('success_rate', 0):.1f}%")
    print(f"   Avg Recommendations per User: {performance.get('average_recommendations_per_user', 0):.1f}")
    print(f"   Avg Recommendation Score: {performance.get('average_recommendation_score', 0):.3f}")
    
    return {
        "insights": insights,
        "recommendations": recommendations,
        "performance": performance
    }


def integration_best_practices():
    """Display best practices for integrating the database-enabled system."""
    print(f"\n🎯 DATABASE INTEGRATION BEST PRACTICES")
    print("=" * 50)
    
    practices = [
        "1. Database Connection Management:",
        "   - Use connection pooling for production",
        "   - Implement proper error handling and retries",
        "   - Monitor database performance and optimize queries",
        "",
        "2. Data Migration:",
        "   - Run migration scripts to set up database schema",
        "   - Import existing JSON data using provided migration tools",
        "   - Validate data integrity after migration",
        "",
        "3. Performance Optimization:",
        "   - Index frequently queried columns (user_id, campaign_id)",
        "   - Cache user profiles for frequently accessed users",
        "   - Use batch operations for bulk data processing",
        "",
        "4. Error Handling:",
        "   - Implement graceful degradation when database is unavailable",
        "   - Log errors for monitoring and debugging",
        "   - Provide meaningful error messages to users",
        "",
        "5. Security:",
        "   - Use parameterized queries to prevent SQL injection",
        "   - Implement proper authentication and authorization",
        "   - Encrypt sensitive data in the database",
        "",
        "6. Monitoring:",
        "   - Track recommendation quality metrics",
        "   - Monitor system performance and response times",
        "   - Set up alerts for system failures"
    ]
    
    for practice in practices:
        print(practice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Recommendation System (DB) Runner")
    parser.add_argument("--user-id", help="User ID to analyze/recommend for")
    parser.add_argument(
        "--action",
        choices=["demo", "insights", "recommend", "perf"],
        default="demo",
        help="What to run: full demo, insights, recommend, or perf evaluation",
    )
    parser.add_argument("--db-type", help="Database type: sqlite, postgresql, mysql")
    parser.add_argument("--db-host", help="Database host")
    parser.add_argument("--db-port", type=int, help="Database port")
    parser.add_argument("--db-name", help="Database name")
    parser.add_argument("--db-user", help="Database username")
    parser.add_argument("--db-password", help="Database password")
    parser.add_argument("--sqlite-path", help="SQLite file path when using sqlite")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip table creation/init (use existing DB as-is)",
    )

    args = parser.parse_args()

    if args.skip_init:
        os.environ["DB_INIT_ENABLED"] = "0"

    db_type = args.db_type or os.getenv("DB_TYPE", "sqlite")
    db_host = args.db_host or os.getenv("DB_HOST", "localhost")
    db_port = args.db_port or int(os.getenv("DB_PORT", "5432"))
    db_name = args.db_name or os.getenv("DB_NAME", "recommendation_db")
    db_user = args.db_user or os.getenv("DB_USER", "")
    db_password = args.db_password or os.getenv("DB_PASSWORD", "")
    sqlite_path = args.sqlite_path or os.getenv("SQLITE_PATH", "data/recommendation.db")

    # Sync provided args into environment so components using env stay consistent
    os.environ["DB_TYPE"] = db_type
    os.environ["DB_HOST"] = db_host
    os.environ["DB_PORT"] = str(db_port)
    os.environ["DB_NAME"] = db_name
    os.environ["DB_USER"] = db_user
    os.environ["DB_PASSWORD"] = db_password
    os.environ["SQLITE_PATH"] = sqlite_path

    print("🚀 Starting Enhanced Recommendation System (DB) with provided configuration...")
    print(f"   DB Type: {db_type}")
    if db_type == "sqlite":
        print(f"   SQLite Path: {sqlite_path}")
    else:
        print(f"   Host: {db_host}:{db_port} | DB: {db_name} | User: {db_user}")
    print(f"   Skip Init: {'YES' if os.getenv('DB_INIT_ENABLED', '1') == '0' else 'NO'}")

    config = DatabaseConfig(
        db_type=db_type,
        host=db_host,
        port=db_port,
        database=db_name,
        username=db_user,
        password=db_password,
        sqlite_path=sqlite_path if db_type == "sqlite" else None,
    )
    db_manager = DatabaseManager(config)
    system = EnhancedRecommendationSystemDB(db_manager)

    if args.action == "demo":
        demo_results = demonstrate_integration()
        integration_best_practices()
        print(f"\n💾 Demo completed successfully!")
        print(
            f"🎯 The enhanced recommendation system with database is ready for production integration!"
        )
    elif args.action == "insights":
        if not args.user_id:
            raise SystemExit("--user-id is required for insights action")
        insights = system.get_user_insights(args.user_id)
        if "error" in insights:
            print(f"❌ {insights['error']}")
        else:
            print(f"✅ Insights for user {args.user_id}:")
            print(f"   Behavioral Type: {insights.get('behavioral_profile', {}).get('behavioral_type', 'Unknown')}")
            print(
                f"   Engagement Level: {insights.get('behavioral_profile', {}).get('engagement_level', {}).get('level', 'Unknown')}"
            )
            print(f"   Total Donations: {insights.get('total_donations', 0)}")
            print(
                f"   Primary Categories: {insights.get('preference_profile', {}).get('category_semantics', {}).get('primary_categories', [])}"
            )
    elif args.action == "recommend":
        if not args.user_id:
            raise SystemExit("--user-id is required for recommend action")
        recommendations = system.get_enhanced_recommendations(
            args.user_id, max_recommendations=10, include_explanations=True
        )
        recs = recommendations.get("recommendations", {}).get("recommendations", [])
        print(f"✅ Generated {len(recs)} recommendations for user {args.user_id}")
        for i, rec in enumerate(recs[:5], 1):
            print(f"\n   {i}. {rec.get('title', 'Unknown Title')}")
            print(f"      Score: {rec.get('final_score', 0):.3f}")
            print(f"      Category: {rec.get('category', 'Unknown')}")
            print(f"      Target: IDR {rec.get('target_amount', 0):,}")
    elif args.action == "perf":
        perf = system.evaluate_system_performance()
        print(f"✅ System Performance:")
        print(f"   Users Tested: {perf.get('users_tested', 0)}")
        print(f"   Success Rate: {perf.get('success_rate', 0):.1f}%")
        print(
            f"   Avg Recommendations per User: {perf.get('average_recommendations_per_user', 0):.1f}"
        )
        print(
            f"   Avg Recommendation Score: {perf.get('average_recommendation_score', 0):.3f}"
        )