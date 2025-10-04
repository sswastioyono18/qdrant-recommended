"""
Smart Filtering System

This module implements intelligent filtering that uses donation history
to predict user preferences and filter undonated projects accordingly.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import statistics
from advanced_profiler import AdvancedUserProfiler
from donation_analyzer import DonationAnalyzer


class SmartProjectFilter:
    """Intelligent filtering system for project recommendations."""
    
    def __init__(self, donations_path: str = "../../data/donations.json", 
                 projects_path: str = "../../data/campaigns.json"):
        """Initialize the smart filter with data paths."""
        self.profiler = AdvancedUserProfiler(donations_path, projects_path)
        self.analyzer = DonationAnalyzer(donations_path, projects_path)
        self.donations = self.analyzer.donations
        # Keep campaigns for JSON compatibility
        self.campaigns = self.analyzer.campaigns
        self.campaign_lookup = self.analyzer.campaign_lookup
    
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
        undonated_projects = [
            project for project in self.campaigns  # Keep campaigns for JSON compatibility
            if str(project['id']) not in donated_project_ids
        ]
        
        if not undonated_projects:
            return {"error": "No undonated projects available"}
        
        # Apply smart filtering
        filtered_projects = self._apply_smart_filters(
            user_profile, undonated_projects, include_reasoning
        )
        
        # Sort by relevance score
        filtered_projects.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Limit results
        top_projects = filtered_projects[:max_projects]
        
        return {
            "user_id": user_id,
            "total_undonated": len(undonated_projects),
            "filtered_count": len(top_projects),
            "profile_summary": self._create_profile_summary(user_profile),
            "projects": top_projects,
            "filtering_strategy": self._get_filtering_strategy(user_profile)
        }
    
    def _get_user_donated_projects(self, user_id: str) -> Set[str]:
        """Get set of project IDs that user has donated to."""
        return {
            str(donation['campaign_id']) for donation in self.donations  # Keep campaign_id for JSON compatibility
            if str(donation['user_id']) == str(user_id)
        }
    
    def _apply_smart_filters(self, user_profile: Dict[str, Any], 
                           projects: List[Dict], 
                           include_reasoning: bool) -> List[Dict[str, Any]]:
        """Apply intelligent filtering based on user profile."""
        filtered_projects = []
        
        for project in projects:
            # Calculate relevance score
            relevance_score, reasoning = self._calculate_project_relevance(
                user_profile, project, include_reasoning
            )
            
            # Apply threshold filtering
            if relevance_score >= self._get_relevance_threshold(user_profile):
                project_data = {
                    "campaign_id": project['id'],  # Keep campaign_id for JSON compatibility
                    "title": project['title'],
                    "category_name": project.get('category_name', 'Unknown'),
                    "target_amount": project.get('target_amount', 0),
                    "country": project.get('country', 'Unknown'),
                    "is_active": project.get('is_active', True),
                    "relevance_score": relevance_score
                }
                
                if include_reasoning:
                    project_data["reasoning"] = reasoning
                
                filtered_projects.append(project_data)
        
        return filtered_projects
    
    def _calculate_project_relevance(self, user_profile: Dict[str, Any], 
                                    project: Dict[str, Any], 
                                    include_reasoning: bool) -> Tuple[float, Optional[Dict]]:
        """Calculate how relevant a project is to the user."""
        relevance_factors = {}
        reasoning = {} if include_reasoning else None
        
        # Category relevance
        category_score = self._calculate_category_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['category'] = category_score
        
        # Amount relevance (based on user's donation patterns)
        amount_score = self._calculate_amount_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['amount'] = amount_score
        
        # Geographic relevance
        geographic_score = self._calculate_geographic_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['geographic'] = geographic_score
        
        # Project size relevance
        size_score = self._calculate_size_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['size'] = size_score
        
        # Risk tolerance relevance
        risk_score = self._calculate_risk_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['risk'] = risk_score
        
        # Social influence (similar users' preferences)
        social_score = self._calculate_social_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['social'] = social_score
        
        # Predictive relevance (based on predicted interests)
        predictive_score = self._calculate_predictive_relevance(
            user_profile, project, reasoning
        )
        relevance_factors['predictive'] = predictive_score
        
        # Calculate weighted overall score
        weights = self._get_relevance_weights(user_profile)
        overall_score = sum(
            relevance_factors[factor] * weights[factor] 
            for factor in relevance_factors
        )
        
        if reasoning:
            reasoning['factor_scores'] = relevance_factors
            reasoning['weights'] = weights
            reasoning['overall_calculation'] = f"Weighted sum of factors: {overall_score:.3f}"
        
        return overall_score, reasoning
    
    def _calculate_category_relevance(self, user_profile: Dict[str, Any], 
                                    project: Dict[str, Any], 
                                    reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on category preferences."""
        project_category = project.get('category_name', '').lower()
        category_prefs = user_profile.get('category_preferences', {}).get('preferences', {})
        
        # Direct category match
        direct_score = 0.0
        for pref_category, stats in category_prefs.items():
            if pref_category.lower() in project_category or project_category in pref_category.lower():
                direct_score = stats.get('combined_score', 0.0)
                break
        
        # Semantic category match (from advanced profiling)
        semantic_score = 0.0
        semantic_groups = user_profile.get('preference_profile', {}).get(
            'category_insights', {}
        ).get('semantic_groups', {})
        
        for group, stats in semantic_groups.items():
            if self._category_matches_semantic_group(project_category, group):
                semantic_score = max(semantic_score, stats.get('preference_strength', 0.0))
        
        # Combine scores
        category_score = max(direct_score, semantic_score * 0.8)  # Semantic slightly lower weight
        
        if reasoning:
            reasoning['category'] = {
                "project_category": project_category,
                "direct_match_score": direct_score,
                "semantic_match_score": semantic_score,
                "final_score": category_score,
                "explanation": f"Category '{project_category}' relevance based on user preferences"
            }
        
        return category_score
    
    def _category_matches_semantic_group(self, category: str, group: str) -> bool:
        """Check if category matches semantic group."""
        group_mappings = {
            "health_medical": ["health", "medical", "healthcare", "hospital", "treatment"],
            "education_development": ["education", "school", "learning", "scholarship", "training"],
            "social_welfare": ["social", "welfare", "community", "family", "children"],
            "emergency_disaster": ["emergency", "disaster", "relief", "urgent", "crisis"],
            "religious_spiritual": ["religious", "spiritual", "mosque", "church", "faith"]
        }
        
        keywords = group_mappings.get(group, [])
        return any(keyword in category.lower() for keyword in keywords)
    
    def _calculate_amount_relevance(self, user_profile: Dict[str, Any], 
                                  project: Dict[str, Any], 
                                  reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on user's donation amount patterns."""
        target_amount = project.get('target_amount', 0)
        
        # Get user's predicted next donation amount
        predictive_profile = user_profile.get('predictive_profile', {})
        predicted_amount = predictive_profile.get('predicted_next_amount', {})
        
        if 'predicted_amount' not in predicted_amount:
            amount_score = 0.5  # Neutral score if no prediction
        else:
            user_amount = predicted_amount['predicted_amount']
            
            # Calculate what percentage of project target user might contribute
            if target_amount > 0:
                contribution_ratio = user_amount / target_amount
                
                # Score based on meaningful contribution (sweet spot around 0.1% to 1%)
                if 0.001 <= contribution_ratio <= 0.01:  # 0.1% to 1%
                    amount_score = 1.0
                elif 0.0001 <= contribution_ratio <= 0.001:  # 0.01% to 0.1%
                    amount_score = 0.8
                elif 0.01 <= contribution_ratio <= 0.1:  # 1% to 10%
                    amount_score = 0.9
                elif contribution_ratio > 0.1:  # > 10%
                    amount_score = 0.6  # Might be too small project
                else:  # < 0.01%
                    amount_score = 0.4  # Might be too large project
            else:
                amount_score = 0.5
        
        if reasoning:
            reasoning['amount'] = {
                "target_amount": target_amount,
                "predicted_user_amount": predicted_amount.get('predicted_amount', 'Unknown'),
                "contribution_ratio": user_amount / target_amount if target_amount > 0 and 'predicted_amount' in predicted_amount else 'Unknown',
                "score": amount_score,
                "explanation": "Relevance based on meaningful contribution potential"
            }
        
        return amount_score
    
    def _calculate_geographic_relevance(self, user_profile: Dict[str, Any], 
                                      project: Dict[str, Any], 
                                      reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on geographic preferences."""
        project_country = project.get('country', 'Unknown')
        
        geographic_prefs = user_profile.get('preference_profile', {}).get(
            'geographic_preferences', {}
        ).get('country_preferences', {})
        
        if project_country in geographic_prefs:
            # User has donated to this country before
            country_stats = geographic_prefs[project_country]
            geographic_score = country_stats.get('frequency_preference', 0.0)
        else:
            # New country - give moderate score based on user's geographic diversity
            diversity = user_profile.get('preference_profile', {}).get(
                'geographic_preferences', {}
            ).get('geographic_diversity', 0)
            
            if diversity > 2:  # User donates to multiple countries
                geographic_score = 0.6  # Open to new countries
            else:
                geographic_score = 0.3  # Prefers familiar countries
        
        if reasoning:
            reasoning['geographic'] = {
                "project_country": project_country,
                "user_country_experience": project_country in geographic_prefs,
                "geographic_diversity": user_profile.get('preference_profile', {}).get(
                    'geographic_preferences', {}
                ).get('geographic_diversity', 0),
                "score": geographic_score,
                "explanation": "Geographic relevance based on user's country preferences"
            }
        
        return geographic_score
    
    def _calculate_size_relevance(self, user_profile: Dict[str, Any], 
                                project: Dict[str, Any], 
                                reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on project size preferences."""
        target_amount = project.get('target_amount', 0)
        
        # Categorize project size
        if target_amount < 10000000:  # < 10M
            project_size = "small"
        elif target_amount < 50000000:  # < 50M
            project_size = "medium"
        elif target_amount < 200000000:  # < 200M
            project_size = "large"
        else:
            project_size = "mega"
        
        # Get user's size preferences
        size_prefs = user_profile.get('preference_profile', {}).get(
            'size_preferences', {}
        ).get('size_preferences', {})
        
        preferred_size = user_profile.get('preference_profile', {}).get(
            'size_preferences', {}
        ).get('preferred_size')
        
        if project_size in size_prefs:
            size_score = size_prefs[project_size]
        else:
            size_score = 0.3  # Default for unknown size preference
        
        # Boost if it matches preferred size
        if project_size == preferred_size:
            size_score = min(size_score * 1.2, 1.0)
        
        if reasoning:
            reasoning['size'] = {
                "project_size": project_size,
                "target_amount": target_amount,
                "user_preferred_size": preferred_size,
                "size_preference_score": size_prefs.get(project_size, 0.3),
                "final_score": size_score,
                "explanation": f"Project size '{project_size}' relevance to user preferences"
            }
        
        return size_score
    
    def _calculate_risk_relevance(self, user_profile: Dict[str, Any], 
                                project: Dict[str, Any], 
                                reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on user's risk tolerance."""
        # Calculate project risk factors
        target_amount = project.get('target_amount', 0)
        is_active = project.get('is_active', True)
        
        # Risk assessment
        target_risk = min(target_amount / 100000000, 1.0)  # Higher target = higher risk
        status_risk = 0.0 if is_active else 0.5  # Inactive = higher risk
        project_risk = (target_risk + status_risk) / 2
        
        # User's risk tolerance
        risk_tolerance = user_profile.get('behavioral_profile', {}).get(
            'risk_tolerance', {}
        ).get('tolerance_level', 'moderate')
        
        # Match risk level to tolerance
        if risk_tolerance == 'conservative':
            if project_risk < 0.3:
                risk_score = 1.0
            elif project_risk < 0.6:
                risk_score = 0.6
            else:
                risk_score = 0.2
        elif risk_tolerance == 'moderate':
            if project_risk < 0.6:
                risk_score = 1.0
            else:
                risk_score = 0.7
        else:  # aggressive
            risk_score = 1.0  # Comfortable with any risk level
        
        if reasoning:
            reasoning['risk'] = {
                "project_risk_level": project_risk,
                "user_risk_tolerance": risk_tolerance,
                "target_amount": target_amount,
                "is_active": is_active,
                "score": risk_score,
                "explanation": f"Risk compatibility: {risk_tolerance} user vs {project_risk:.2f} risk project"
            }
        
        return risk_score
    
    def _calculate_social_relevance(self, user_profile: Dict[str, Any], 
                                  project: Dict[str, Any], 
                                  reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on similar users' preferences."""
        similar_users = user_profile.get('social_profile', {}).get('similar_users', [])
        
        if not similar_users:
            social_score = 0.5  # Neutral if no similar users
        else:
            # Check if similar users have donated to this project
            project_id = str(project['id'])
            similar_donations = 0
            total_weight = 0
            
            for similar_user in similar_users:
                similarity_weight = similar_user.get('similarity_score', 0)
                similar_user_id = similar_user.get('user_id')
                
                if similar_user_id and similar_user_id in self.donations:
                    user_projects = set(
                        str(d['campaign_id']) for d in self.donations[similar_user_id]
                    )
                    if project_id in user_projects:
                        similar_donations += similarity_weight
                    total_weight += similarity_weight
            
            if total_weight > 0:
                social_score = similar_donations / total_weight
            else:
                social_score = 0.5
        
        if reasoning:
            reasoning['social'] = {
                "similar_users_count": len(similar_users),
                "social_validation_score": social_score,
                "explanation": "Relevance based on similar users' donation patterns"
            }
        
        return social_score
    
    def _calculate_predictive_relevance(self, user_profile: Dict[str, Any], 
                                      project: Dict[str, Any], 
                                      reasoning: Optional[Dict]) -> float:
        """Calculate relevance based on predictive insights."""
        predictive_profile = user_profile.get('predictive_profile', {})
        
        # Check predicted category interests
        predicted_interests = predictive_profile.get(
            'predicted_category_interests', {}
        ).get('predicted_interests', {})
        
        project_category = project.get('category_name', '').lower()
        
        predictive_score = 0.0
        for category, interest_score in predicted_interests.items():
            if category.lower() in project_category or project_category in category.lower():
                predictive_score = max(predictive_score, interest_score)
        
        # Factor in donation likelihood
        likelihood = predictive_profile.get('donation_likelihood', {}).get('likelihood_score', 0.5)
        predictive_score = predictive_score * likelihood
        
        if reasoning:
            reasoning['predictive'] = {
                "predicted_category_match": predictive_score > 0,
                "donation_likelihood": likelihood,
                "final_score": predictive_score,
                "explanation": "Relevance based on AI-predicted future interests"
            }
        
        return predictive_score
    
    def _get_relevance_weights(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Get weights for different relevance factors based on user profile."""
        # Base weights
        weights = {
            'category': 0.25,
            'amount': 0.15,
            'geographic': 0.10,
            'size': 0.15,
            'risk': 0.10,
            'social': 0.15,
            'predictive': 0.10
        }
        
        # Adjust weights based on profile characteristics
        profile_completeness = user_profile.get('profile_completeness', 0.5)
        
        # If profile is incomplete, rely more on basic factors
        if profile_completeness < 0.5:
            weights['category'] = 0.4
            weights['amount'] = 0.2
            weights['social'] = 0.05
            weights['predictive'] = 0.05
        
        # If user has strong behavioral patterns, weight them more
        behavioral_type = user_profile.get('behavioral_profile', {}).get('behavioral_type')
        if behavioral_type in ['loyal_consistent', 'loyal_variable']:
            weights['category'] += 0.1
            weights['predictive'] += 0.05
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_relevance_threshold(self, user_profile: Dict[str, Any]) -> float:
        """Get minimum relevance threshold based on user profile."""
        # Base threshold
        threshold = 0.3
        
        # Adjust based on profile characteristics
        engagement_level = user_profile.get('behavioral_profile', {}).get(
            'engagement_level', {}
        ).get('level', 'medium')
        
        if engagement_level == 'high':
            threshold = 0.4  # Higher standards for engaged users
        elif engagement_level == 'low':
            threshold = 0.2  # Lower threshold for new users
        
        # Adjust based on recommendation confidence
        confidence = user_profile.get('recommendation_confidence', 0.5)
        if confidence > 0.8:
            threshold += 0.1  # Higher threshold when confident
        elif confidence < 0.3:
            threshold -= 0.1  # Lower threshold when uncertain
        
        return max(0.1, min(0.6, threshold))  # Keep within reasonable bounds
    
    def _create_profile_summary(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the user profile for display."""
        return {
            "behavioral_type": user_profile.get('behavioral_profile', {}).get('behavioral_type'),
            "engagement_level": user_profile.get('behavioral_profile', {}).get(
                'engagement_level', {}
            ).get('level'),
            "risk_tolerance": user_profile.get('behavioral_profile', {}).get(
                'risk_tolerance', {}
            ).get('tolerance_level'),
            "community_position": user_profile.get('social_profile', {}).get('community_position'),
            "donation_likelihood": user_profile.get('predictive_profile', {}).get(
                'donation_likelihood', {}
            ).get('likelihood_level'),
            "profile_completeness": user_profile.get('profile_completeness'),
            "recommendation_confidence": user_profile.get('recommendation_confidence')
        }
    
    def _get_filtering_strategy(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the filtering strategy used for this user."""
        weights = self._get_relevance_weights(user_profile)
        threshold = self._get_relevance_threshold(user_profile)
        
        # Identify primary filtering factors
        primary_factors = [
            factor for factor, weight in weights.items() 
            if weight > 0.2
        ]
        
        return {
            "primary_factors": primary_factors,
            "relevance_threshold": threshold,
            "weights": weights,
            "strategy_description": self._describe_strategy(user_profile, primary_factors)
        }
    
    def _describe_strategy(self, user_profile: Dict[str, Any], 
                          primary_factors: List[str]) -> str:
        """Create a human-readable description of the filtering strategy."""
        behavioral_type = user_profile.get('behavioral_profile', {}).get('behavioral_type')
        engagement = user_profile.get('behavioral_profile', {}).get(
            'engagement_level', {}
        ).get('level')
        
        if behavioral_type == 'loyal_consistent' and engagement == 'high':
            return f"Personalized filtering for loyal user focusing on {', '.join(primary_factors)}"
        elif engagement == 'low':
            return f"Exploratory filtering for new user emphasizing {', '.join(primary_factors)}"
        else:
            return f"Balanced filtering approach prioritizing {', '.join(primary_factors)}"


def example_usage():
    """Demonstrate the smart filtering capabilities."""
    filter_system = SmartProjectFilter()
    
    print("=== SMART PROJECT FILTERING DEMO ===\n")
    
    user_id = "1001"
    print(f"🎯 Smart Filtered Projects for User {user_id}:")
    print("=" * 60)
    
    # Get smart filtered projects
    result = filter_system.get_smart_filtered_projects(
        user_id, max_projects=5, include_reasoning=True
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Display summary
    print(f"📊 Filtering Summary:")
    print(f"  • Total undonated projects: {result['total_undonated']}")
    print(f"  • Projects after smart filtering: {result['filtered_count']}")
    
    # Display profile summary
    profile = result['profile_summary']
    print(f"\n👤 User Profile Summary:")
    print(f"  • Behavioral Type: {profile['behavioral_type']}")
    print(f"  • Engagement Level: {profile['engagement_level']}")
    print(f"  • Risk Tolerance: {profile['risk_tolerance']}")
    print(f"  • Community Position: {profile['community_position']}")
    print(f"  • Recommendation Confidence: {profile['recommendation_confidence']:.1%}")
    
    # Display filtering strategy
    strategy = result['filtering_strategy']
    print(f"\n🧠 Filtering Strategy:")
    print(f"  • Strategy: {strategy['strategy_description']}")
    print(f"  • Primary Factors: {', '.join(strategy['primary_factors'])}")
    print(f"  • Relevance Threshold: {strategy['relevance_threshold']:.2f}")
    
    # Display top projects
    print(f"\n🎯 Top Recommended Projects:")
    for i, project in enumerate(result['projects'], 1):
        print(f"\n{i}. {project['title']}")
        print(f"   Category: {project['category_name']}")
        print(f"   Target: IDR {project['target_amount']:,}")
        print(f"   Country: {project['country']}")
        print(f"   Relevance Score: {project['relevance_score']:.3f}")
        
        if 'reasoning' in project:
            reasoning = project['reasoning']
            print(f"   🔍 Key Factors:")
            
            # Show top contributing factors
            factor_scores = reasoning.get('factor_scores', {})
            top_factors = sorted(
                factor_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for factor, score in top_factors:
                print(f"     • {factor.title()}: {score:.3f}")
    
    print(f"\n" + "=" * 60)
    
    # Compare with another user
    print(f"\n🔄 Comparison with User 2002:")
    result_2002 = filter_system.get_smart_filtered_projects(
        "2002", max_projects=3, include_reasoning=False
    )
    
    if "error" not in result_2002:
        print(f"  • User 2002 filtered projects: {result_2002['filtered_count']}")
        print(f"  • Strategy: {result_2002['filtering_strategy']['strategy_description']}")
        print(f"  • Top project: {result_2002['projects'][0]['title'] if result_2002['projects'] else 'None'}")


if __name__ == "__main__":
    example_usage()