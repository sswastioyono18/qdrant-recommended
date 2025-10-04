"""
Comprehensive Recommendation Engine - Database Version

This module combines multiple recommendation approaches using database:
1. AI-powered semantic similarity (existing system)
2. Smart filtering based on donation history
3. Advanced user profiling
4. Social and collaborative filtering
5. Predictive modeling

The engine provides a unified, sophisticated recommendation system.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import statistics

from .smart_filter_db import SmartProjectFilterDB
from .advanced_profiler_db import AdvancedUserProfilerDB
from .donation_analyzer_db import DatabaseDonationAnalyzer
from .database import DatabaseManager
from .models import ProjectRepository, DonationRepository, UserRepository


# Module logger for structured debug output
logger = logging.getLogger(__name__)


class ComprehensiveRecommendationEngineDB:
    """
    Advanced recommendation engine that combines multiple approaches
    for superior project recommendations using database.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the comprehensive recommendation engine with database manager."""
        self.db_manager = db_manager
        
        # Initialize component systems
        self.smart_filter = SmartProjectFilterDB(db_manager)
        self.profiler = AdvancedUserProfilerDB(db_manager)
        self.analyzer = DatabaseDonationAnalyzer(auto_init_db=True)
        
        # Initialize repositories
        self.project_repo = ProjectRepository(db_manager)
        self.donation_repo = DonationRepository(db_manager)
        self.user_repo = UserRepository(db_manager)
        
        # Scoring weights for different approaches
        self.scoring_weights = {
            "smart_filter": 0.30,      # Smart filtering based on user profile
            "semantic_similarity": 0.25,  # AI-powered semantic matching
            "collaborative": 0.20,     # Collaborative filtering
            "popularity": 0.15,        # Campaign popularity
            "diversity": 0.10          # Diversity optimization
        }
    
    def get_comprehensive_recommendations(self, user_id: str, 
                                        max_recommendations: int = 10,
                                        include_explanations: bool = True,
                                        diversity_factor: float = 0.3) -> Dict[str, Any]:
        """
        Get comprehensive recommendations using multiple approaches.
        
        Args:
            user_id: User ID to get recommendations for
            max_recommendations: Maximum number of recommendations to return
            include_explanations: Whether to include detailed explanations
            diversity_factor: Factor for diversity optimization (0.0 to 1.0)
        
        Returns:
            Dictionary containing recommendations with scores and explanations
        """
        # Get user profile
        user_profile = self.profiler.create_comprehensive_profile(user_id)
        
        if "error" in user_profile:
            return {"error": f"Could not create profile for user {user_id}"}
        
        # Get projects user hasn't donated to
        donated_project_ids = self._get_user_donated_projects(user_id)
        undonated_projects = self.analyzer.get_undonated_projects(user_id)
        
        if not undonated_projects:
            return {
                "recommendations": [],
                "user_profile_summary": self._create_user_summary(user_profile),
                "strategy_description": self._describe_strategy(user_profile, diversity_factor),
                "message": "No undonated projects available"
            }
        
        # Calculate comprehensive scores for all projects
        scored_projects = self._calculate_comprehensive_scores(
            user_id, user_profile, undonated_projects, include_explanations
        )
        
        # Apply diversity optimization
        if diversity_factor > 0:
            scored_projects = self._apply_diversity_optimization(
                scored_projects, diversity_factor
            )
        
        # Sort by final score and limit results
        scored_projects.sort(key=lambda x: x["final_score"], reverse=True)
        recommendations = scored_projects[:max_recommendations]
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(user_profile, recommendations)
        
        return {
            "recommendations": recommendations,
            "total_candidates": len(undonated_projects),
            "user_profile_summary": self._create_user_summary(user_profile),
            "strategy_description": self._describe_strategy(user_profile, diversity_factor),
            "scoring_weights": self.scoring_weights,
            "performance_metrics": performance_metrics,
            "diversity_factor_applied": diversity_factor
        }
    
    def _get_user_donated_projects(self, user_id: str) -> Set[str]:
        """Get set of project IDs the user has already donated to."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        return {str(donation.project_id) for donation in user_donations}
    
    def _calculate_comprehensive_scores(self, user_id: str, 
                                      user_profile: Dict[str, Any],
                                      projects: List[Dict],
                                      include_explanations: bool) -> List[Dict[str, Any]]:
        """Calculate comprehensive scores using multiple approaches."""
        scored_projects = []
        
        for project in projects:
            # Get scores from different approaches
            smart_filter_score, smart_explanation = self._get_smart_filter_score(
                user_profile, project, include_explanations
            )
            
            semantic_score, semantic_explanation = self._get_semantic_similarity_score(
                user_id, project, include_explanations
            )
            
            collaborative_score, collaborative_explanation = self._get_collaborative_score(
                user_id, user_profile, project, include_explanations
            )
            
            popularity_score, popularity_explanation = self._get_popularity_score(
                project, include_explanations
            )
            
            # Calculate weighted final score
            component_scores = {
                "smart_filter": smart_filter_score,
                "semantic_similarity": semantic_score,
                "collaborative": collaborative_score,
                "popularity": popularity_score
            }
            
            final_score = sum(
                score * self.scoring_weights[component] 
                for component, score in component_scores.items()
            )
            
            # Create project result
            project_result = {
                **project,
                "final_score": final_score,
                "component_scores": component_scores,
                "score_breakdown": self._create_score_breakdown(component_scores, self.scoring_weights)
            }
            
            # Add explanations if requested
            if include_explanations:
                project_result["explanations"] = {
                    "smart_filter": smart_explanation,
                    "semantic_similarity": semantic_explanation,
                    "collaborative": collaborative_explanation,
                    "popularity": popularity_explanation
                }
            
            scored_projects.append(project_result)
        
        return scored_projects
    
    def _get_smart_filter_score(self, user_profile: Dict[str, Any], 
                              project: Dict[str, Any],
                              include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score from smart filtering approach."""
        # Use the smart filter's relevance calculation
        relevance_score, reasoning = self.smart_filter._calculate_project_relevance(
            user_profile, project, include_explanations
        )
        
        explanation = None
        if include_explanations and reasoning:
            explanation = {
                "approach": "Smart filtering based on user profile and preferences",
                "relevance_score": relevance_score,
                "key_factors": list(reasoning.get("component_scores", {}).keys()),
                "reasoning_summary": f"Score based on user's behavioral type and preferences"
            }
        
        return relevance_score, explanation
    
    def _get_semantic_similarity_score(self, user_id: str, 
                                     project: Dict[str, Any],
                                     include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score from semantic similarity approach."""
        # Helper to normalize category-like values to lowercase strings
        def _normalize_category(val: Any) -> str:
            if val is None:
                return ""
            try:
                s = str(val)
            except Exception:
                return ""
            return s.lower()
        # Get user's donation history for semantic analysis
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        if not user_donations:
            return 0.5, {"approach": "Semantic similarity", "note": "No donation history for comparison"}
        
        # Get categories from user's donation history
        user_categories = []
        for donation in user_donations:
            project_data = self.project_repo.get_by_id(donation.project_id)
            if project_data:
                user_categories.append(_normalize_category(getattr(project_data, "category", "")))
        
        # Calculate semantic similarity based on category overlap
        project_category = _normalize_category(project.get("category", ""))
        
        # Direct category match
        if project_category in user_categories:
            semantic_score = 0.9
        else:
            # Semantic category matching
            semantic_score = self._calculate_semantic_category_similarity(
                project_category, user_categories
            )
        
        explanation = None
        if include_explanations:
            explanation = {
                "approach": "Semantic similarity based on donation history",
                "project_category": project_category,
                "user_categories": list(set(user_categories)),
                "similarity_score": semantic_score,
                "reasoning": "Based on category overlap and semantic relationships"
            }
        
        return semantic_score, explanation
    
    def _calculate_semantic_category_similarity(self, campaign_category: str, 
                                              user_categories: List[str]) -> float:
        """Calculate semantic similarity between categories."""
        # Define semantic groups
        semantic_groups = {
            "education": ["scholarship", "school", "learning", "student", "education"],
            "health": ["medical", "healthcare", "hospital", "treatment", "health"],
            "disaster": ["emergency", "relief", "natural disaster", "crisis", "disaster"],
            "poverty": ["hunger", "basic needs", "social welfare", "poverty"],
            "environment": ["nature", "conservation", "climate", "environment"],
            "children": ["kids", "children", "orphan", "child welfare"]
        }
        
        # Find semantic group for campaign category
        campaign_group = None
        for group, keywords in semantic_groups.items():
            if campaign_category in keywords:
                campaign_group = group
                break
        
        if not campaign_group:
            return 0.3  # Default similarity for unknown categories
        
        # Check if any user categories are in the same semantic group
        for user_category in user_categories:
            for group, keywords in semantic_groups.items():
                if user_category in keywords and group == campaign_group:
                    return 0.7  # High similarity for same semantic group
        
        return 0.3  # Low similarity for different semantic groups
    
    def _get_collaborative_score(self, user_id: str, 
                               user_profile: Dict[str, Any],
                               project: Dict[str, Any],
                               include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score from collaborative filtering approach."""
        # Get similar users from profile
        social_profile = user_profile.get("social_profile", {})
        similar_users = social_profile.get("similar_users", [])

        project_id = str(project.get("id", ""))
        logger.debug(
            "Collaborative check: user_id=%s project_id=%s similar_users_count=%d",
            user_id,
            project_id,
            len(similar_users),
        )

        if not similar_users:
            logger.debug(
                "Collaborative check: No similar users for user_id=%s; default score applied",
                user_id,
            )
            return 0.5, {"approach": "Collaborative filtering", "note": "No similar users found"}

        # Check how many similar users donated to this project
        similar_user_donations = 0
        total_similar_users = min(10, len(similar_users))  # Consider top 10 similar users
        
        for similar_user in similar_users[:total_similar_users]:
            similar_user_id = similar_user.get("user_id", "")
            if similar_user_id:
                user_donations = self.donation_repo.get_by_user_id(similar_user_id)
                donated_to_project = any(str(d.project_id) == project_id for d in user_donations)
                logger.debug(
                    "Collaborative check: similar_user_id=%s donations_fetched=%d donated_to_project=%s",
                    similar_user_id,
                    len(user_donations),
                    donated_to_project,
                )
                if donated_to_project:
                    similar_user_donations += 1
        
        # Calculate collaborative score
        if total_similar_users > 0:
            collaborative_score = similar_user_donations / total_similar_users
        else:
            collaborative_score = 0.5

        logger.debug(
            "Collaborative result: user_id=%s project_id=%s analyzed=%d donated=%d score=%.3f",
            user_id,
            project_id,
            total_similar_users,
            similar_user_donations,
            collaborative_score,
        )

        explanation = None
        if include_explanations:
            explanation = {
                "approach": "Collaborative filtering based on similar users",
                "similar_users_analyzed": total_similar_users,
                "similar_users_who_donated": similar_user_donations,
                "collaborative_score": collaborative_score,
                "reasoning": f"{similar_user_donations} out of {total_similar_users} similar users donated to this project"
        }
        
        return collaborative_score, explanation
    
    def _get_popularity_score(self, project: Dict[str, Any],
                            include_explanations: bool) -> Tuple[float, Optional[Dict]]:
        """Get score based on project popularity."""
        project_id = str(project.get("id", ""))
        
        # Get all donations for this project
        project_donations = self.donation_repo.get_by_project_id(project_id)
        donation_count = len(project_donations)
        
        # Calculate total amount raised
        total_raised = sum(donation.amount for donation in project_donations)
        target_amount = project.get("target_amount", 1)
        
        # Calculate popularity metrics
        funding_ratio = total_raised / target_amount if target_amount > 0 else 0
        
        # Normalize donation count (assuming max 100 donations for normalization)
        normalized_donation_count = min(donation_count / 100, 1.0)
        
        # Combine metrics for popularity score
        popularity_score = (funding_ratio * 0.6 + normalized_donation_count * 0.4)
        popularity_score = min(popularity_score, 1.0)  # Cap at 1.0
        
        explanation = None
        if include_explanations:
            explanation = {
                "approach": "Campaign popularity based on donations and funding",
                "donation_count": donation_count,
                "total_raised": total_raised,
                "target_amount": target_amount,
                "funding_ratio": funding_ratio,
                "popularity_score": popularity_score,
                "reasoning": f"Based on {donation_count} donations and {funding_ratio:.1%} funding progress"
            }
        
        return popularity_score, explanation
    
    def _apply_diversity_optimization(self, recommendations: List[Dict[str, Any]], 
                                    diversity_factor: float) -> List[Dict[str, Any]]:
        """Apply diversity optimization to avoid too many similar recommendations."""
        if diversity_factor <= 0 or len(recommendations) <= 1:
            return recommendations
        
        # Group recommendations by category
        category_groups = defaultdict(list)
        for rec in recommendations:
            raw_cat = rec.get("category", "unknown")
            try:
                category = str(raw_cat).lower()
            except Exception:
                category = "unknown"
            category_groups[category].append(rec)
        
        # Apply diversity penalty to over-represented categories
        for category, recs in category_groups.items():
            if len(recs) > 1:
                # Sort by score within category
                recs.sort(key=lambda x: x["final_score"], reverse=True)
                
                # Apply increasing penalty to lower-ranked items in same category
                for i, rec in enumerate(recs[1:], 1):  # Skip the top item in each category
                    penalty = diversity_factor * (i * 0.1)  # Increasing penalty
                    rec["final_score"] *= (1 - penalty)
                    rec["diversity_penalty_applied"] = penalty
        
        return recommendations
    
    def _create_score_breakdown(self, component_scores: Dict[str, float], 
                              weights: Dict[str, float]) -> Dict[str, Any]:
        """Create a detailed breakdown of how the final score was calculated."""
        weighted_scores = {}
        total_weighted = 0
        
        for component, score in component_scores.items():
            weight = weights.get(component, 0)
            weighted_score = score * weight
            weighted_scores[component] = {
                "raw_score": score,
                "weight": weight,
                "weighted_score": weighted_score
            }
            total_weighted += weighted_score
        
        return {
            "component_breakdown": weighted_scores,
            "final_weighted_score": total_weighted
        }
    
    def _create_user_summary(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the user profile for context."""
        behavioral_profile = user_profile.get("behavioral_profile", {})
        preference_profile = user_profile.get("preference_profile", {})
        
        return {
            "behavioral_type": behavioral_profile.get("behavioral_type", "unknown"),
            "engagement_level": behavioral_profile.get("engagement_level", {}).get("level", "unknown"),
            "risk_tolerance": behavioral_profile.get("risk_tolerance", {}).get("tolerance", "unknown"),
            "primary_categories": preference_profile.get("category_semantics", {}).get("primary_categories", []),
            "total_donations": user_profile.get("total_donations", 0),
            "average_amount": user_profile.get("avg_amount", 0),
            "donation_frequency": behavioral_profile.get("consistency", {}).get("frequency", "unknown")
        }
    
    def _describe_strategy(self, user_profile: Dict[str, Any], 
                         diversity_factor: float) -> Dict[str, Any]:
        """Describe the recommendation strategy used."""
        behavioral_type = user_profile.get("behavioral_profile", {}).get("behavioral_type", "unknown")
        
        strategy_descriptions = {
            "champion": "High-engagement strategy with emphasis on smart filtering and collaborative signals",
            "supporter": "Balanced approach combining user preferences with popular campaigns",
            "steady_giver": "Consistency-focused strategy prioritizing familiar categories and reliable campaigns",
            "occasional_donor": "Broad exploration strategy with semantic similarity emphasis",
            "new_donor": "Discovery-oriented approach using popularity and diversity optimization"
        }
        
        base_strategy = strategy_descriptions.get(
            behavioral_type, 
            "Standard multi-approach recommendation strategy"
        )
        
        diversity_note = ""
        if diversity_factor > 0.5:
            diversity_note = " with high diversity optimization"
        elif diversity_factor > 0.2:
            diversity_note = " with moderate diversity optimization"
        elif diversity_factor > 0:
            diversity_note = " with light diversity optimization"
        
        return {
            "strategy_description": base_strategy + diversity_note,
            "primary_approaches": ["smart_filter", "semantic_similarity", "collaborative"],
            "diversity_factor": diversity_factor,
            "scoring_weights": self.scoring_weights
        }
    
    def _calculate_performance_metrics(self, user_profile: Dict[str, Any], 
                                     recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for the recommendations."""
        if not recommendations:
            return {"error": "No recommendations to analyze"}
        
        # Score distribution analysis
        scores = [rec["final_score"] for rec in recommendations]
        
        # Category diversity analysis (safe normalization)
        categories = [
            str(rec.get("category", "unknown") if rec.get("category") is not None else "unknown").lower()
            for rec in recommendations
        ]
        unique_categories = len(set(categories))
        
        # Amount range analysis
        amounts = [rec.get("target_amount", 0) for rec in recommendations]
        
        return {
            "score_statistics": {
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "score_range": max(scores) - min(scores),
                "top_score": max(scores)
            },
            "diversity_metrics": {
                "unique_categories": unique_categories,
                "total_recommendations": len(recommendations),
                "category_diversity_ratio": unique_categories / len(recommendations)
            },
            "amount_range": {
                "min_target": min(amounts) if amounts else 0,
                "max_target": max(amounts) if amounts else 0,
                "avg_target": statistics.mean(amounts) if amounts else 0
            },
            "recommendation_quality": self._assess_recommendation_quality(user_profile, recommendations)
        }
    
    def _assess_recommendation_quality(self, user_profile: Dict[str, Any], 
                                     recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of recommendations based on user profile."""
        behavioral_type = user_profile.get("behavioral_profile", {}).get("behavioral_type", "unknown")
        
        # Quality thresholds based on user type
        quality_thresholds = {
            "champion": 0.7,
            "supporter": 0.6,
            "steady_giver": 0.6,
            "occasional_donor": 0.5,
            "new_donor": 0.4
        }
        
        threshold = quality_thresholds.get(behavioral_type, 0.5)
        high_quality_count = sum(1 for rec in recommendations if rec["final_score"] >= threshold)
        
        return {
            "quality_threshold": threshold,
            "high_quality_recommendations": high_quality_count,
            "quality_ratio": high_quality_count / len(recommendations),
            "assessment": "Excellent" if high_quality_count / len(recommendations) >= 0.8 else
                         "Good" if high_quality_count / len(recommendations) >= 0.6 else
                         "Fair" if high_quality_count / len(recommendations) >= 0.4 else "Needs Improvement"
        }


def example_usage():
    """Example usage of the ComprehensiveRecommendationEngineDB."""
    from .database import DatabaseConfig, DatabaseManager
    
    # Initialize database
    config = DatabaseConfig()
    db_manager = DatabaseManager(config)
    
    # Create comprehensive recommendation engine
    engine = ComprehensiveRecommendationEngineDB(db_manager)
    
    # Get comprehensive recommendations for a user
    user_id = "1001"
    result = engine.get_comprehensive_recommendations(
        user_id=user_id,
        max_recommendations=5,
        include_explanations=True,
        diversity_factor=0.3
    )
    
    print(f"Comprehensive Recommendations for User {user_id}:")
    print(f"Total candidates analyzed: {result.get('total_candidates', 0)}")
    print(f"User type: {result.get('user_profile_summary', {}).get('behavioral_type', 'Unknown')}")
    print(f"Strategy: {result.get('strategy_description', {}).get('strategy_description', 'Unknown')}")
    
    for i, rec in enumerate(result.get('recommendations', []), 1):
        print(f"\n{i}. {rec.get('title', 'Unknown Title')}")
        print(f"   Final Score: {rec.get('final_score', 0):.3f}")
        print(f"   Category: {rec.get('category', 'Unknown')}")
        print(f"   Target: IDR {rec.get('target_amount', 0):,}")
        
        # Show component scores
        component_scores = rec.get('component_scores', {})
        print(f"   Component Scores:")
        for component, score in component_scores.items():
            print(f"     {component}: {score:.3f}")


if __name__ == "__main__":
    example_usage()