"""
Comprehensive Recommendation Engine

This module combines multiple recommendation approaches:
1. AI-powered semantic similarity (existing system)
2. Smart filtering based on donation history
3. Advanced user profiling
4. Social and collaborative filtering
5. Predictive modeling

The engine provides a unified, sophisticated recommendation system.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import statistics
import sys
import os

# Add the parent directory to the path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_filter import SmartProjectFilter
from advanced_profiler import AdvancedUserProfiler
from donation_analyzer import DonationAnalyzer


class ComprehensiveRecommendationEngine:
    """
    Advanced recommendation engine that combines multiple approaches
    for superior project recommendations.
    """
    
    def __init__(self, donations_path: str = "../../data/donations.json", 
                 projects_path: str = "../../data/campaigns.json"):
        """Initialize the comprehensive recommendation engine."""
        self.donations_path = donations_path
        self.projects_path = projects_path
        
        # Initialize component systems
        self.smart_filter = SmartProjectFilter(donations_path, projects_path)
        self.profiler = AdvancedUserProfiler(donations_path, projects_path)
        self.analyzer = DonationAnalyzer(donations_path, projects_path)
        
        # Load data
        self.donations = self.analyzer.donations
        self.projects = self.analyzer.campaigns  # Note: keeping campaigns for JSON compatibility
        self.project_lookup = self.analyzer.campaign_lookup  # Note: keeping campaign_lookup for JSON compatibility
        
        # Initialize recommendation weights
        self.recommendation_weights = {
            'smart_filter': 0.35,      # Smart filtering based on history
            'semantic_similarity': 0.25,  # AI semantic matching
            'collaborative': 0.20,     # Similar users' preferences
            'popularity': 0.10,        # Campaign popularity/trending
            'diversity': 0.10          # Recommendation diversity
        }
    
    def get_comprehensive_recommendations(self, user_id: str, 
                                        max_recommendations: int = 10,
                                        include_explanations: bool = True,
                                        diversity_factor: float = 0.3) -> Dict[str, Any]:
        """
        Get comprehensive recommendations combining multiple approaches.
        
        Args:
            user_id: User ID to get recommendations for
            max_recommendations: Maximum number of recommendations
            include_explanations: Whether to include detailed explanations
            diversity_factor: How much to prioritize diversity (0.0 to 1.0)
        
        Returns:
            Comprehensive recommendation results with scores and explanations
        """
        print(f"🚀 Generating comprehensive recommendations for user {user_id}...")
        
        # Step 1: Get user profile and validate
        user_profile = self.profiler.create_comprehensive_profile(user_id)
        if "error" in user_profile:
            return {"error": f"Could not create profile for user {user_id}"}
        
        # Step 2: Get projects user hasn't donated to
        donated_project_ids = self._get_user_donated_projects(user_id)
        undonated_projects = [
            project for project in self.projects 
            if str(project['id']) not in donated_project_ids
        ]
        
        if not undonated_projects:
            return {"error": "No undonated projects available"}
        
        print(f"📊 Found {len(undonated_projects)} undonated projects to analyze...")
        
        # Step 3: Apply multiple recommendation approaches
        recommendation_scores = self._calculate_comprehensive_scores(
            user_id, user_profile, undonated_projects, include_explanations
        )
        
        # Step 4: Apply diversity optimization
        if diversity_factor > 0:
            recommendation_scores = self._apply_diversity_optimization(
                recommendation_scores, diversity_factor
            )
        
        # Step 5: Sort and limit results
        recommendation_scores.sort(key=lambda x: x['final_score'], reverse=True)
        top_recommendations = recommendation_scores[:max_recommendations]
        
        # Step 6: Generate final result
        result = {
            "user_id": user_id,
            "total_undonated_projects": len(undonated_projects),
            "recommendations_generated": len(top_recommendations),
            "user_profile_summary": self._create_user_summary(user_profile),
            "recommendation_strategy": self._describe_strategy(user_profile, diversity_factor),
            "recommendations": top_recommendations,
            "performance_metrics": self._calculate_performance_metrics(
                user_profile, top_recommendations
            )
        }
        
        print(f"✅ Generated {len(top_recommendations)} comprehensive recommendations")
        return result
    
    def _get_user_donated_projects(self, user_id: str) -> Set[str]:
        """Get set of project IDs user has already donated to."""
        user_donations = self.donations.get(user_id, [])
        return set(str(donation['campaign_id']) for donation in user_donations)  # Note: keeping campaign_id for JSON compatibility
    
    def _calculate_comprehensive_scores(self, user_id: str, 
                                      user_profile: Dict[str, Any],
                                      projects: List[Dict],
                                      include_explanations: bool) -> List[Dict[str, Any]]:
        """Calculate comprehensive scores using multiple approaches."""
        scored_projects = []
        
        print(f"🧮 Calculating scores using {len(self.recommendation_weights)} approaches...")
        
        for project in projects:
            project_scores = {}
            explanations = {} if include_explanations else None
            
            # 1. Smart Filter Score
            smart_score, smart_explanation = self._get_smart_filter_score(
                user_profile, project, include_explanations
            )
            project_scores['smart_filter'] = smart_score
            if explanations:
                explanations['smart_filter'] = smart_explanation
            
            # 2. Semantic Similarity Score
            semantic_score, semantic_explanation = self._get_semantic_similarity_score(
                user_id, project, include_explanations
            )
            project_scores['semantic_similarity'] = semantic_score
            if explanations:
                explanations['semantic_similarity'] = semantic_explanation
            
            # 3. Collaborative Filtering Score
            collaborative_score, collaborative_explanation = self._get_collaborative_score(
                user_id, user_profile, project, include_explanations
            )
            project_scores['collaborative'] = collaborative_score
            if explanations:
                explanations['collaborative'] = collaborative_explanation
            
            # 4. Popularity Score
            popularity_score, popularity_explanation = self._get_popularity_score(
                project, include_explanations
            )
            project_scores['popularity'] = popularity_score
            if explanations:
                explanations['popularity'] = popularity_explanation
            
            # 5. Calculate weighted final score
            final_score = sum(
                project_scores[approach] * self.recommendation_weights[approach]
                for approach in project_scores
            )
            
            # Create project recommendation object
            project_rec = {
                "campaign_id": project['id'],  # Note: keeping campaign_id for JSON compatibility
                "title": project['title'],
                "category_name": project.get('category_name', 'Unknown'),
                "target_amount": project.get('target_amount', 0),
                "country": project.get('country', 'Unknown'),
                "is_active": project.get('is_active', True),
                "final_score": final_score,
                "component_scores": project_scores,
                "score_weights": self.recommendation_weights.copy()
            }
            
            if explanations:
                project_rec["explanations"] = explanations
                project_rec["score_breakdown"] = self._create_score_breakdown(
                    project_scores, self.recommendation_weights
                )
            
            scored_projects.append(project_rec)
        
        return scored_projects
    
    def _get_smart_filter_score(self, user_profile: Dict[str, Any], 
                              project: Dict[str, Any],
                              include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score from smart filtering system."""
        try:
            # Use smart filter to get relevance scores  
            results = self.smart_filter.get_smart_filtered_campaigns(
                user_profile, [project], max_results=1
            )
            
            if results and len(results) > 0:
                score = results[0].get('relevance_score', 0.0)
                explanation = results[0].get('reasoning', {}) if include_explanations else None
                return score, explanation
            else:
                return 0.0, {"error": "No smart filter results"} if include_explanations else None
            
        except Exception as e:
            print(f"⚠️ Smart filter error for project {project.get('id')}: {e}")
            return 0.3, {"error": str(e)} if include_explanations else None
    
    def _get_semantic_similarity_score(self, user_id: str, 
                                     project: Dict[str, Any],
                                     include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score from semantic similarity (simulated - would use actual AI system)."""
        # This would integrate with your existing AI recommendation system
        # For now, we'll simulate based on project characteristics
        
        try:
            # Simulate semantic similarity based on project content
            title_words = project.get('title', '').lower().split()
            description_words = project.get('description', '').lower().split()
            
            # Get user's historical preferences
            user_donations = self.donations.get(user_id, [])
            if not user_donations:
                semantic_score = 0.4  # Default for new users
            else:
                # Simulate semantic matching based on historical projects
                historical_projects = [
                    self.project_lookup.get(str(d['campaign_id']), {})  # Note: keeping campaign_id for JSON compatibility
                    for d in user_donations
                ]
                
                # Simple keyword overlap simulation
                user_keywords = set()
                for hist_project in historical_projects:
                    if hist_project:
                        user_keywords.update(hist_project.get('title', '').lower().split())
                        user_keywords.update(hist_project.get('description', '').lower().split())
                
                project_keywords = set(title_words + description_words)
                
                if user_keywords and project_keywords:
                    overlap = len(user_keywords.intersection(project_keywords))
                    semantic_score = min(overlap / 10.0, 1.0)  # Normalize
                else:
                    semantic_score = 0.3
            
            explanation = None
            if include_explanations:
                explanation = {
                    "score": semantic_score,
                    "description": "AI semantic similarity to user's historical preferences",
                    "method": "Keyword overlap simulation (would use actual embeddings)",
                    "user_keyword_count": len(user_keywords) if 'user_keywords' in locals() else 0
                }
            
            return semantic_score, explanation
            
        except Exception as e:
            print(f"⚠️ Semantic similarity error for project {project.get('id')}: {e}")
            return 0.3, {"error": str(e)} if include_explanations else None
    
    def _get_collaborative_score(self, user_id: str, 
                               user_profile: Dict[str, Any],
                               project: Dict[str, Any],
                               include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score from collaborative filtering (similar users)."""
        try:
            similar_users = user_profile.get('social_profile', {}).get('similar_users', [])
            project_id = str(project['id'])
            
            if not similar_users:
                collaborative_score = 0.4  # Neutral score
                similar_user_count = 0
                donation_rate = 0.0
            else:
                # Check how many similar users donated to this project
                donations_by_similar = 0
                total_weight = 0
                
                for similar_user in similar_users:
                    similarity_weight = similar_user.get('similarity_score', 0)
                    similar_user_id = similar_user.get('user_id')
                    
                    if similar_user_id and similar_user_id in self.donations:
                        user_projects = set(
                            str(d['campaign_id']) for d in self.donations[similar_user_id]  # Note: keeping campaign_id for JSON compatibility
                        )
                        if project_id in user_projects:
                            donations_by_similar += similarity_weight
                        total_weight += similarity_weight
                
                if total_weight > 0:
                    donation_rate = donations_by_similar / total_weight
                    collaborative_score = donation_rate
                else:
                    collaborative_score = 0.4
                    donation_rate = 0.0
                
                similar_user_count = len(similar_users)
            
            explanation = None
            if include_explanations:
                explanation = {
                    "score": collaborative_score,
                    "description": "Preference based on similar users' donation patterns",
                    "similar_users_analyzed": similar_user_count,
                    "donation_rate_by_similar_users": f"{donation_rate:.1%}",
                    "method": "Weighted collaborative filtering"
                }
            
            return collaborative_score, explanation
            
        except Exception as e:
            print(f"⚠️ Collaborative filtering error for campaign {campaign.get('id')}: {e}")
            return 0.3, {"error": str(e)} if include_explanations else None
    
    def _get_popularity_score(self, project: Dict[str, Any],
                            include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score based on project popularity and trending."""
        try:
            project_id = str(project['id'])
            
            # Count total donations to this project
            total_donations = 0
            total_donors = 0
            total_amount = 0
            
            for user_id, user_donations in self.donations.items():
                for donation in user_donations:
                    if str(donation['campaign_id']) == project_id:  # Note: keeping campaign_id for JSON compatibility
                        total_donations += 1
                        total_amount += donation.get('amount', 0)
                
                # Count unique donors
                user_project_ids = set(str(d['campaign_id']) for d in user_donations)  # Note: keeping campaign_id for JSON compatibility
                if project_id in user_project_ids:
                    total_donors += 1
            
            # Calculate popularity metrics
            max_donations = max(
                sum(1 for user_donations in self.donations.values() 
                    for donation in user_donations 
                    if str(donation['campaign_id']) == cid)  # Note: keeping campaign_id for JSON compatibility
                for cid in set(
                    str(d['campaign_id'])  # Note: keeping campaign_id for JSON compatibility
                    for user_donations in self.donations.values() 
                    for d in user_donations
                )
            ) if self.donations else 1
            
            # Normalize popularity score
            donation_popularity = total_donations / max_donations if max_donations > 0 else 0
            donor_popularity = total_donors / len(self.donations) if self.donations else 0
            
            # Combine metrics
            popularity_score = (donation_popularity * 0.6 + donor_popularity * 0.4)
            
            # Add small boost for active projects
            if project.get('is_active', True):
                popularity_score *= 1.1
            
            popularity_score = min(popularity_score, 1.0)  # Cap at 1.0
            
            explanation = None
            if include_explanations:
                explanation = {
                    "score": popularity_score,
                    "description": "Project popularity and community engagement",
                    "total_donations": total_donations,
                    "unique_donors": total_donors,
                    "total_amount_raised": total_amount,
                    "is_active": project.get('is_active', True)
                }
            
            return popularity_score, explanation
            
        except Exception as e:
            print(f"⚠️ Popularity scoring error for project {project.get('id')}: {e}")
            return 0.3, {"error": str(e)} if include_explanations else None
    
    def _apply_diversity_optimization(self, recommendations: List[Dict[str, Any]], 
                                    diversity_factor: float) -> List[Dict[str, Any]]:
        """Apply diversity optimization to avoid too similar recommendations."""
        if diversity_factor <= 0 or len(recommendations) <= 1:
            return recommendations
        
        print(f"🎨 Applying diversity optimization (factor: {diversity_factor})...")
        
        # Group by category for diversity
        category_groups = defaultdict(list)
        for rec in recommendations:
            category = rec.get('category_name', 'Unknown')
            category_groups[category].append(rec)
        
        # Apply diversity penalty for over-representation
        for category, recs in category_groups.items():
            if len(recs) > 1:
                # Sort by score within category
                recs.sort(key=lambda x: x['final_score'], reverse=True)
                
                # Apply increasing penalty to lower-ranked items in same category
                for i, rec in enumerate(recs[1:], 1):  # Skip first (best) in category
                    penalty = diversity_factor * (i * 0.1)  # Increasing penalty
                    rec['final_score'] *= (1 - penalty)
                    rec['diversity_penalty_applied'] = penalty
        
        return recommendations
    
    def _create_score_breakdown(self, component_scores: Dict[str, float], 
                              weights: Dict[str, float]) -> Dict[str, Any]:
        """Create detailed score breakdown for explanations."""
        breakdown = {}
        total_weighted = 0
        
        for component, score in component_scores.items():
            weight = weights.get(component, 0)
            weighted_score = score * weight
            total_weighted += weighted_score
            
            breakdown[component] = {
                "raw_score": score,
                "weight": weight,
                "weighted_contribution": weighted_score,
                "percentage_of_total": (weighted_score / total_weighted * 100) if total_weighted > 0 else 0
            }
        
        breakdown["total_score"] = total_weighted
        return breakdown
    
    def _create_user_summary(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise user summary for the recommendation result."""
        return {
            "behavioral_type": user_profile.get('behavioral_profile', {}).get('behavioral_type'),
            "engagement_level": user_profile.get('behavioral_profile', {}).get(
                'engagement_level', {}
            ).get('level'),
            "risk_tolerance": user_profile.get('behavioral_profile', {}).get(
                'risk_tolerance', {}
            ).get('tolerance_level'),
            "donation_frequency": user_profile.get('behavioral_profile', {}).get(
                'donation_frequency', {}
            ).get('frequency_level'),
            "preferred_categories": list(
                user_profile.get('category_preferences', {}).get('preferences', {}).keys()
            )[:3],
            "profile_completeness": user_profile.get('profile_completeness'),
            "recommendation_confidence": user_profile.get('recommendation_confidence')
        }
    
    def _describe_strategy(self, user_profile: Dict[str, Any], 
                         diversity_factor: float) -> Dict[str, Any]:
        """Describe the recommendation strategy used."""
        behavioral_type = user_profile.get('behavioral_profile', {}).get('behavioral_type')
        engagement = user_profile.get('behavioral_profile', {}).get(
            'engagement_level', {}
        ).get('level')
        
        # Determine primary strategy
        if behavioral_type == 'loyal_consistent' and engagement == 'high':
            strategy_type = "Personalized Deep Matching"
            description = "Leveraging extensive user history for highly personalized recommendations"
        elif engagement == 'low':
            strategy_type = "Exploratory Broad Matching"
            description = "Balanced approach for users with limited history"
        elif behavioral_type in ['loyal_variable', 'occasional_large']:
            strategy_type = "Adaptive Pattern Matching"
            description = "Adapting to user's variable donation patterns"
        else:
            strategy_type = "Comprehensive Multi-Signal"
            description = "Combining all available signals for optimal recommendations"
        
        return {
            "strategy_type": strategy_type,
            "description": description,
            "primary_signals": [
                signal for signal, weight in self.recommendation_weights.items() 
                if weight > 0.2
            ],
            "diversity_optimization": diversity_factor > 0,
            "weights_used": self.recommendation_weights.copy()
        }
    
    def _calculate_performance_metrics(self, user_profile: Dict[str, Any], 
                                     recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for the recommendations."""
        if not recommendations:
            return {"error": "No recommendations to analyze"}
        
        # Score distribution
        scores = [rec['final_score'] for rec in recommendations]
        
        # Category diversity
        categories = [rec.get('category_name', 'Unknown') for rec in recommendations]
        unique_categories = len(set(categories))
        
        # Confidence metrics
        confidence = user_profile.get('recommendation_confidence', 0.5)
        
        return {
            "average_score": statistics.mean(scores),
            "score_range": max(scores) - min(scores),
            "category_diversity": unique_categories,
            "total_categories_available": len(set(
                project.get('category_name', 'Unknown') for project in self.projects
            )),
            "recommendation_confidence": confidence,
            "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        }


def example_usage():
    """Demonstrate the comprehensive recommendation engine."""
    print("=== COMPREHENSIVE RECOMMENDATION ENGINE DEMO ===\n")
    
    engine = ComprehensiveRecommendationEngine()
    
    user_id = "1001"
    print(f"🎯 Comprehensive Recommendations for User {user_id}:")
    print("=" * 70)
    
    # Get comprehensive recommendations
    result = engine.get_comprehensive_recommendations(
        user_id, 
        max_recommendations=5, 
        include_explanations=True,
        diversity_factor=0.3
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Display summary
    print(f"\n📊 Recommendation Summary:")
    print(f"  • Total undonated projects analyzed: {result['total_undonated_projects']}")
    print(f"  • Final recommendations: {result['recommendations_generated']}")
    
    # Display user profile
    profile = result['user_profile_summary']
    print(f"\n👤 User Profile:")
    print(f"  • Behavioral Type: {profile['behavioral_type']}")
    print(f"  • Engagement: {profile['engagement_level']}")
    print(f"  • Risk Tolerance: {profile['risk_tolerance']}")
    print(f"  • Donation Frequency: {profile['donation_frequency']}")
    print(f"  • Preferred Categories: {', '.join(profile['preferred_categories'])}")
    print(f"  • Profile Completeness: {profile['profile_completeness']:.1%}")
    
    # Display strategy
    strategy = result['recommendation_strategy']
    print(f"\n🧠 Recommendation Strategy:")
    print(f"  • Type: {strategy['strategy_type']}")
    print(f"  • Description: {strategy['description']}")
    print(f"  • Primary Signals: {', '.join(strategy['primary_signals'])}")
    print(f"  • Diversity Optimization: {'Yes' if strategy['diversity_optimization'] else 'No'}")
    
    # Display performance metrics
    metrics = result['performance_metrics']
    print(f"\n📈 Performance Metrics:")
    print(f"  • Average Score: {metrics['average_score']:.3f}")
    print(f"  • Score Range: {metrics['score_range']:.3f}")
    print(f"  • Category Diversity: {metrics['category_diversity']}/{metrics['total_categories_available']}")
    print(f"  • Confidence Level: {metrics['confidence_level']}")
    
    # Display top recommendations
    print(f"\n🎯 Top Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Final Score: {rec['final_score']:.3f}")
        print(f"   Category: {rec['category_name']}")
        print(f"   Target: IDR {rec['target_amount']:,}")
        
        if 'score_breakdown' in rec:
            breakdown = rec['score_breakdown']
            print(f"   📊 Score Components:")
            for component, details in breakdown.items():
                if component != 'total_score':
                    print(f"     • {component.replace('_', ' ').title()}: "
                          f"{details['weighted_contribution']:.3f} "
                          f"({details['percentage_of_total']:.1f}%)")
        
        if 'explanations' in rec:
            print(f"   💡 Key Insights:")
            for approach, explanation in rec['explanations'].items():
                if isinstance(explanation, dict) and 'description' in explanation:
                    print(f"     • {approach.replace('_', ' ').title()}: "
                          f"{explanation['description']}")
    
    print(f"\n" + "=" * 70)
    
    # Quick comparison with another user
    print(f"\n🔄 Quick Comparison with User 2002:")
    result_2002 = engine.get_comprehensive_recommendations(
        "2002", max_recommendations=3, include_explanations=False
    )
    
    if "error" not in result_2002:
        print(f"  • User 2002 recommendations: {result_2002['recommendations_generated']}")
        print(f"  • Strategy: {result_2002['recommendation_strategy']['strategy_type']}")
        print(f"  • Top recommendation: {result_2002['recommendations'][0]['title'] if result_2002['recommendations'] else 'None'}")
        print(f"  • Average score: {result_2002['performance_metrics']['average_score']:.3f}")


if __name__ == "__main__":
    example_usage()