"""
Smart Filtering System - Database Version

This module implements intelligent filtering that uses donation history
to predict user preferences and filter undonated campaigns accordingly.
Uses database instead of JSON files for data retrieval.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import statistics
from .advanced_profiler_db import AdvancedUserProfilerDB
from .donation_analyzer_db import DatabaseDonationAnalyzer
from .database import DatabaseManager
from .models import ProjectRepository, DonationRepository, UserRepository


class SmartProjectFilterDB:
    """Intelligent filtering system for project recommendations using database."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the smart filter with database manager."""
        self.db_manager = db_manager
        self.profiler = AdvancedUserProfilerDB(db_manager)
        self.analyzer = DatabaseDonationAnalyzer(auto_init_db=True)
        self.project_repo = ProjectRepository(db_manager)
        self.donation_repo = DonationRepository(db_manager)
        self.user_repo = UserRepository(db_manager)
    
    def get_smart_filtered_projects(self, user_id: str, 
                                   max_projects: int = 20,
                                   include_reasoning: bool = True) -> Dict[str, Any]:
        """
        Get intelligently filtered projects for a user based on their profile.
        
        Args:
            user_id: User ID to get recommendations for
            max_projects: Maximum number of projects to return
            include_reasoning: Whether to include reasoning for each recommendation
        
        Returns:
            Dictionary containing filtered projects with scores and reasoning
        """
        # Get user's comprehensive profile
        user_profile = self.profiler.create_comprehensive_profile(user_id)
        
        if "error" in user_profile:
            return {"error": f"Could not create profile for user {user_id}"}
        
        # Get projects user hasn't donated to
        donated_project_ids = self._get_user_donated_projects(user_id)
        undonated_projects = self.analyzer.get_undonated_projects(user_id)
        
        if not undonated_projects:
            return {
                "projects": [],
                "total_filtered": 0,
                "user_profile_summary": self._create_profile_summary(user_profile),
                "filtering_strategy": self._get_filtering_strategy(user_profile),
                "message": "No undonated projects available"
            }
        
        # Apply smart filtering
        filtered_projects = self._apply_smart_filters(
            user_profile, undonated_projects, include_reasoning
        )
        
        # Limit results
        limited_projects = filtered_projects[:max_projects]
        
        return {
            "projects": limited_projects,
            "total_filtered": len(filtered_projects),
            "total_available": len(undonated_projects),
            "user_profile_summary": self._create_profile_summary(user_profile),
            "filtering_strategy": self._get_filtering_strategy(user_profile),
            "relevance_threshold": self._get_relevance_threshold(user_profile),
            "filtering_performance": {
                "projects_processed": len(undonated_projects),
                "projects_passed_filter": len(filtered_projects),
                "filter_efficiency": len(filtered_projects) / len(undonated_projects) if undonated_projects else 0
            }
        }
    
    def _get_user_donated_projects(self, user_id: str) -> Set[str]:
        """Get set of project IDs the user has already donated to."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        return {str(donation.project_id) for donation in user_donations}
    
    def _apply_smart_filters(self, user_profile: Dict[str, Any], 
                           projects: List[Dict], 
                           include_reasoning: bool) -> List[Dict[str, Any]]:
        """Apply intelligent filtering based on user profile."""
        filtered_projects = []
        relevance_threshold = self._get_relevance_threshold(user_profile)
        
        for project in projects:
            relevance_score, reasoning = self._calculate_project_relevance(
                user_profile, project, include_reasoning
            )
            
            if relevance_score >= relevance_threshold:
                project_result = {
                    **project,
                    "relevance_score": relevance_score,
                    "filter_passed": True
                }
                
                if include_reasoning and reasoning:
                    project_result["reasoning"] = reasoning
                
                filtered_projects.append(project_result)
        
        # Sort by relevance score (descending)
        filtered_projects.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return filtered_projects
    
    def _calculate_project_relevance(self, user_profile: Dict[str, Any], 
                                    project: Dict[str, Any], 
                                    include_reasoning: bool) -> Tuple[float, Optional[Dict]]:
        """Calculate how relevant a project is to the user."""
        reasoning = {} if include_reasoning else None
        
        # Get relevance weights based on user profile
        weights = self._get_relevance_weights(user_profile)
        
        # Calculate individual relevance scores
        category_score = self._calculate_category_relevance(user_profile, project, reasoning)
        amount_score = self._calculate_amount_relevance(user_profile, project, reasoning)
        geographic_score = self._calculate_geographic_relevance(user_profile, project, reasoning)
        size_score = self._calculate_size_relevance(user_profile, project, reasoning)
        risk_score = self._calculate_risk_relevance(user_profile, project, reasoning)
        social_score = self._calculate_social_relevance(user_profile, project, reasoning)
        predictive_score = self._calculate_predictive_relevance(user_profile, project, reasoning)
        
        # Calculate weighted total score
        total_score = (
            category_score * weights["category"] +
            amount_score * weights["amount"] +
            geographic_score * weights["geographic"] +
            size_score * weights["size"] +
            risk_score * weights["risk"] +
            social_score * weights["social"] +
            predictive_score * weights["predictive"]
        )
        
        if include_reasoning:
            reasoning.update({
                "total_score": total_score,
                "component_scores": {
                    "category": category_score,
                    "amount": amount_score,
                    "geographic": geographic_score,
                    "size": size_score,
                    "risk": risk_score,
                    "social": social_score,
                    "predictive": predictive_score
                },
                "weights_applied": weights
            })
        
        return total_score, reasoning
    
    def _calculate_category_relevance(self, user_profile: Dict[str, Any], 
                                    project: Dict[str, Any], 
                                    reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on category preferences."""
        # Normalize category to lowercase string safely
        raw_category = project.get("category", "")
        try:
            project_category = str(raw_category).lower()
        except Exception:
            project_category = ""
        
        # Get user's category preferences
        preference_profile = user_profile.get("preference_profile", {})
        category_prefs = preference_profile.get("category_semantics", {}).get("category_preferences", {})
        
        # Direct category match
        direct_score = category_prefs.get(project_category, 0)
        
        # Semantic category matching
        semantic_score = 0
        for user_category, preference in category_prefs.items():
            if self._category_matches_semantic_group(project_category, user_category):
                semantic_score = max(semantic_score, preference * 0.8)  # Slightly lower for semantic match
        
        # Take the higher of direct or semantic match
        final_score = max(direct_score, semantic_score)
        
        if reasoning is not None:
            reasoning["category_analysis"] = {
                "project_category": project_category,
                "direct_match_score": direct_score,
                "semantic_match_score": semantic_score,
                "final_score": final_score,
                "user_category_preferences": category_prefs
            }
        
        return final_score
    
    def _category_matches_semantic_group(self, category: str, group: str) -> bool:
        """Check if categories are semantically related."""
        semantic_groups = {
            "education": ["scholarship", "school", "learning", "student"],
            "health": ["medical", "healthcare", "hospital", "treatment"],
            "disaster": ["emergency", "relief", "natural disaster", "crisis"],
            "poverty": ["hunger", "basic needs", "social welfare"]
        }
        
        for group_name, keywords in semantic_groups.items():
            if group in keywords and category in keywords:
                return True
        return False
    
    def _calculate_amount_relevance(self, user_profile: Dict[str, Any], 
                                  project: Dict[str, Any], 
                                  reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on donation amount patterns."""
        project_target = project.get("target_amount", 0)
        
        # Get user's amount preferences
        behavioral_profile = user_profile.get("behavioral_profile", {})
        engagement_level = behavioral_profile.get("engagement_level", {})
        avg_user_amount = engagement_level.get("average_amount", 0)
        
        # Get predictive profile for next donation prediction
        predictive_profile = user_profile.get("predictive_profile", {})
        next_donation = predictive_profile.get("next_donation_prediction", {})
        predicted_amount = next_donation.get("predicted_amount", avg_user_amount)
        
        # Calculate relevance based on how well the project target aligns with user's giving capacity
        if project_target == 0 or predicted_amount == 0:
            score = 0.5  # Neutral score when data is missing
        else:
            # Calculate what percentage of the target the user might contribute
            contribution_ratio = predicted_amount / project_target
            
            # Optimal range: user can contribute 0.1% to 10% of target
            if 0.001 <= contribution_ratio <= 0.1:
                score = 1.0
            elif 0.0001 <= contribution_ratio <= 0.001:
                score = 0.8
            elif 0.1 < contribution_ratio <= 0.5:
                score = 0.6
            else:
                score = 0.3
        
        if reasoning is not None:
            reasoning["amount_analysis"] = {
                "project_target": project_target,
                "user_average_amount": avg_user_amount,
                "predicted_next_amount": predicted_amount,
                "contribution_ratio": predicted_amount / project_target if project_target > 0 else 0,
                "relevance_score": score
            }
        
        return score
    
    def _calculate_geographic_relevance(self, user_profile: Dict[str, Any], 
                                      project: Dict[str, Any], 
                                      reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on geographic preferences."""
        # Handle None location safely: if value is None, default to empty string
        location_val = project.get("location")
        project_location = (location_val or "").strip().lower()
        
        # Get user's geographic preferences
        preference_profile = user_profile.get("preference_profile", {})
        geo_prefs = preference_profile.get("geographic_preferences", {}).get("location_preferences", {})
        
        # Direct location match
        direct_score = geo_prefs.get(project_location, 0)
        
        # If no direct match, give a small score for geographic diversity
        diversity_score = 0.3 if not direct_score and geo_prefs else 0
        
        final_score = max(direct_score, diversity_score)
        
        if reasoning is not None:
            reasoning["geographic_analysis"] = {
                "project_location": project_location,
                "direct_match_score": direct_score,
                "diversity_bonus": diversity_score,
                "final_score": final_score,
                "user_location_preferences": geo_prefs
            }
        
        return final_score
    
    def _calculate_size_relevance(self, user_profile: Dict[str, Any], 
                                project: Dict[str, Any], 
                                reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on project size preferences."""
        project_target = project.get("target_amount", 0)
        
        # Categorize project size
        if project_target < 10000000:  # < 10M IDR
            project_size = "small"
        elif project_target < 100000000:  # < 100M IDR
            project_size = "medium"
        else:
            project_size = "large"
        
        # Get user's size preferences
        preference_profile = user_profile.get("preference_profile", {})
        size_prefs = preference_profile.get("project_size_preferences", {}).get("size_preferences", {})
        
        score = size_prefs.get(project_size, 0.5)  # Default to neutral if no preference
        
        if reasoning is not None:
            reasoning["size_analysis"] = {
                "project_target_amount": project_target,
                "project_size_category": project_size,
                "user_size_preferences": size_prefs,
                "relevance_score": score
            }
        
        return score
    
    def _calculate_risk_relevance(self, user_profile: Dict[str, Any], 
                                project: Dict[str, Any], 
                                reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on user's risk tolerance."""
        # Calculate project risk score
        project_risk = self._calculate_project_risk_score(project)
        
        # Get user's risk tolerance
        behavioral_profile = user_profile.get("behavioral_profile", {})
        risk_tolerance = behavioral_profile.get("risk_tolerance", {})
        user_risk_score = risk_tolerance.get("average_risk_score", 0.5)
        tolerance_level = risk_tolerance.get("tolerance", "medium")
        
        # Calculate relevance based on risk alignment
        risk_diff = abs(project_risk - user_risk_score)
        
        if risk_diff <= 0.2:
            score = 1.0  # Very good match
        elif risk_diff <= 0.4:
            score = 0.7  # Good match
        elif risk_diff <= 0.6:
            score = 0.4  # Moderate match
        else:
            score = 0.2  # Poor match
        
        if reasoning is not None:
            reasoning["risk_analysis"] = {
                "project_risk_score": project_risk,
                "user_risk_score": user_risk_score,
                "user_risk_tolerance": tolerance_level,
                "risk_difference": risk_diff,
                "relevance_score": score
            }
        
        return score
    
    def _calculate_project_risk_score(self, project: Dict[str, Any]) -> float:
        """Calculate risk score for a project."""
        risk_score = 0.5  # Base score
        
        # Adjust based on project characteristics
        target_amount = project.get("target_amount", 0)
        if target_amount > 100000000:  # 100M IDR
            risk_score += 0.2
        
        category = str(project.get("category", "")).lower()
        high_risk_categories = ['emergency', 'disaster', 'medical']
        if category in high_risk_categories:
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _calculate_social_relevance(self, user_profile: Dict[str, Any], 
                                  project: Dict[str, Any], 
                                  reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on social factors."""
        # Get user's social profile
        social_profile = user_profile.get("social_profile", {})
        similar_users = social_profile.get("similar_users", [])
        
        # Check if similar users have donated to this project
        project_id = str(project.get("id", ""))
        similar_user_donations = 0
        
        for similar_user in similar_users[:5]:  # Check top 5 similar users
            similar_user_id = similar_user.get("user_id", "")
            if similar_user_id:
                user_donations = self.donation_repo.get_by_user_id(similar_user_id)
                if any(str(d.project_id) == project_id for d in user_donations):
                    similar_user_donations += 1
        
        # Calculate social score based on similar user behavior
        if len(similar_users) > 0:
            social_score = similar_user_donations / min(5, len(similar_users))
        else:
            social_score = 0.5  # Neutral when no similar users
        
        if reasoning is not None:
            reasoning["social_analysis"] = {
                "similar_users_count": len(similar_users),
                "similar_users_who_donated": similar_user_donations,
                "social_influence_score": social_score
            }
        
        return social_score
    
    def _calculate_predictive_relevance(self, user_profile: Dict[str, Any], 
                                      project: Dict[str, Any], 
                                      reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on predictive insights."""
        predictive_profile = user_profile.get("predictive_profile", {})
        
        # Get category interest prediction
        category_prediction = predictive_profile.get("category_interest_prediction", {})
        predicted_categories = category_prediction.get("predicted_categories", {})
        
        project_category = str(project.get("category", "")).lower()
        category_interest = predicted_categories.get(project_category, 0.5)
        
        # Get donation likelihood
        donation_likelihood = predictive_profile.get("donation_likelihood", {})
        likelihood_score = donation_likelihood.get("score", 0.5)
        
        # Combine predictive factors
        predictive_score = (category_interest + likelihood_score) / 2
        
        if reasoning is not None:
            reasoning["predictive_analysis"] = {
                "predicted_category_interest": category_interest,
                "donation_likelihood": likelihood_score,
                "combined_predictive_score": predictive_score
            }
        
        return predictive_score
    
    def _get_relevance_weights(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Get relevance weights based on user profile characteristics."""
        # Default weights
        weights = {
            "category": 0.25,
            "amount": 0.20,
            "geographic": 0.15,
            "size": 0.10,
            "risk": 0.10,
            "social": 0.10,
            "predictive": 0.10
        }
        
        # Adjust weights based on user profile
        behavioral_profile = user_profile.get("behavioral_profile", {})
        behavioral_type = behavioral_profile.get("behavioral_type", "")
        
        if behavioral_type == "champion":
            weights["category"] = 0.30  # Champions care more about categories
            weights["social"] = 0.15    # And social influence
        elif behavioral_type == "steady_giver":
            weights["amount"] = 0.25    # Steady givers care about amount consistency
            weights["risk"] = 0.15      # And risk management
        
        return weights
    
    def _get_relevance_threshold(self, user_profile: Dict[str, Any]) -> float:
        """Get relevance threshold based on user profile."""
        behavioral_profile = user_profile.get("behavioral_profile", {})
        behavioral_type = behavioral_profile.get("behavioral_type", "")
        
        # Adjust threshold based on user type
        if behavioral_type == "champion":
            return 0.7  # High threshold for champions
        elif behavioral_type in ["supporter", "steady_giver"]:
            return 0.6  # Medium threshold for regular donors
        elif behavioral_type == "occasional_donor":
            return 0.5  # Lower threshold for occasional donors
        else:
            return 0.4  # Lowest threshold for new donors
    
    def _create_profile_summary(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the user profile for filtering context."""
        behavioral_profile = user_profile.get("behavioral_profile", {})
        preference_profile = user_profile.get("preference_profile", {})
        
        return {
            "behavioral_type": behavioral_profile.get("behavioral_type", "unknown"),
            "engagement_level": behavioral_profile.get("engagement_level", {}).get("level", "unknown"),
            "risk_tolerance": behavioral_profile.get("risk_tolerance", {}).get("tolerance", "unknown"),
            "primary_categories": preference_profile.get("category_semantics", {}).get("primary_categories", []),
            "preferred_campaign_size": preference_profile.get("campaign_size_preferences", {}).get("preferred_size", "unknown"),
            "total_donations": user_profile.get("total_donations", 0),
            "average_amount": user_profile.get("avg_amount", 0)
        }
    
    def _get_filtering_strategy(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get the filtering strategy used for this user."""
        weights = self._get_relevance_weights(user_profile)
        threshold = self._get_relevance_threshold(user_profile)
        
        # Identify primary filtering factors
        primary_factors = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        primary_factor_names = [factor[0] for factor in primary_factors]
        
        return {
            "relevance_threshold": threshold,
            "primary_factors": primary_factor_names,
            "factor_weights": weights,
            "strategy_description": self._describe_strategy(user_profile, primary_factor_names)
        }
    
    def _describe_strategy(self, user_profile: Dict[str, Any], 
                          primary_factors: List[str]) -> str:
        """Describe the filtering strategy in human-readable terms."""
        behavioral_type = user_profile.get("behavioral_profile", {}).get("behavioral_type", "unknown")
        
        strategy_descriptions = {
            "champion": f"High-engagement filtering focusing on {', '.join(primary_factors[:2])}",
            "supporter": f"Balanced filtering emphasizing {', '.join(primary_factors[:2])}",
            "steady_giver": f"Consistency-focused filtering prioritizing {', '.join(primary_factors[:2])}",
            "occasional_donor": f"Broad filtering with emphasis on {primary_factors[0]}",
            "new_donor": f"Exploratory filtering to discover preferences via {primary_factors[0]}"
        }
        
        return strategy_descriptions.get(behavioral_type, f"Standard filtering using {', '.join(primary_factors[:2])}")


def example_usage():
    """Example usage of the SmartProjectFilterDB."""
    from .database import DatabaseConfig, DatabaseManager
    
    # Initialize database
    config = DatabaseConfig()
    db_manager = DatabaseManager(config)
    
    # Create smart filter
    smart_filter = SmartProjectFilterDB(db_manager)
    
    # Get filtered projects for a user
    user_id = "1001"
    result = smart_filter.get_smart_filtered_projects(
        user_id=user_id,
        max_projects=10,
        include_reasoning=True
    )
    
    print(f"Smart Filtered Projects for User {user_id}:")
    print(f"Total projects filtered: {result.get('total_filtered', 0)}")
    print(f"User behavioral type: {result.get('user_profile_summary', {}).get('behavioral_type', 'Unknown')}")
    print(f"Filtering strategy: {result.get('filtering_strategy', {}).get('strategy_description', 'Unknown')}")
    
    for i, project in enumerate(result.get('projects', [])[:3], 1):
        print(f"\n{i}. {project.get('title', 'Unknown Title')}")
        print(f"   Relevance Score: {project.get('relevance_score', 0):.3f}")
        print(f"   Category: {project.get('category', 'Unknown')}")
        if 'reasoning' in project:
            reasoning = project['reasoning']
            print(f"   Top factors: {list(reasoning.get('component_scores', {}).keys())[:3]}")


if __name__ == "__main__":
    example_usage()