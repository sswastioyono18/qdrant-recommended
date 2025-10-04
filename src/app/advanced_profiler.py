"""
Advanced User Profiling System

This module creates sophisticated user profiles that combine multiple signals
from donation history to enable highly personalized recommendations.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter
import statistics
from donation_analyzer import DonationAnalyzer


class AdvancedUserProfiler:
    """Creates comprehensive user profiles for personalized recommendations."""
    
    def __init__(self, donations_path: str = "data/donations.json", 
                 campaigns_path: str = "data/campaigns.json"):
        """Initialize the profiler with data paths."""
        self.analyzer = DonationAnalyzer(donations_path, campaigns_path)
        self.donations = self.analyzer.donations
        self.campaigns = self.analyzer.campaigns
        self.campaign_lookup = self.analyzer.campaign_lookup
    
    def create_comprehensive_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Create a comprehensive user profile that combines multiple dimensions:
        - Behavioral patterns (frequency, consistency, timing)
        - Preference patterns (categories, amounts, geography)
        - Social patterns (similarity to other users)
        - Predictive patterns (likely future behavior)
        """
        base_profile = self.analyzer.get_user_donation_profile(user_id)
        
        if "error" in base_profile:
            return base_profile
        
        # Add advanced profiling layers
        behavioral_profile = self._create_behavioral_profile(user_id)
        preference_profile = self._create_preference_profile(user_id)
        social_profile = self._create_social_profile(user_id)
        predictive_profile = self._create_predictive_profile(user_id)
        
        return {
            **base_profile,
            "behavioral_profile": behavioral_profile,
            "preference_profile": preference_profile,
            "social_profile": social_profile,
            "predictive_profile": predictive_profile,
            "profile_completeness": self._calculate_profile_completeness(base_profile),
            "recommendation_confidence": self._calculate_recommendation_confidence(base_profile)
        }
    
    def _create_behavioral_profile(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's behavioral patterns in donations."""
        user_donations = self.donations.get(user_id, [])
        
        if not user_donations:
            return {"error": "No donations to analyze"}
        
        amounts = [d['amount'] for d in user_donations]
        
        # Donation consistency analysis
        consistency_metrics = self._analyze_donation_consistency(amounts)
        
        # Engagement level analysis
        engagement_level = self._calculate_engagement_level(user_donations)
        
        # Risk tolerance (based on campaign target amounts)
        risk_tolerance = self._analyze_risk_tolerance(user_id, user_donations)
        
        return {
            "consistency_metrics": consistency_metrics,
            "engagement_level": engagement_level,
            "risk_tolerance": risk_tolerance,
            "donation_momentum": self._calculate_donation_momentum(user_donations),
            "behavioral_type": self._classify_behavioral_type(consistency_metrics, engagement_level)
        }
    
    def _create_preference_profile(self, user_id: str) -> Dict[str, Any]:
        """Create detailed preference analysis."""
        user_donations = self.donations.get(user_id, [])
        
        # Enhanced category analysis with semantic grouping
        category_insights = self._analyze_category_semantics(user_id)
        
        # Geographic preferences
        geographic_preferences = self._analyze_geographic_preferences(user_id)
        
        # Campaign size preferences
        size_preferences = self._analyze_campaign_size_preferences(user_id)
        
        # Impact preferences
        impact_preferences = self._analyze_impact_preferences(user_id)
        
        return {
            "category_insights": category_insights,
            "geographic_preferences": geographic_preferences,
            "size_preferences": size_preferences,
            "impact_preferences": impact_preferences,
            "preference_strength": self._calculate_preference_strength(user_donations)
        }
    
    def _create_social_profile(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's position in the donor community."""
        similar_users = self.analyzer.find_similar_users(user_id, top_k=10)
        
        # Cluster analysis
        user_cluster = self._identify_user_cluster(user_id, similar_users)
        
        # Influence analysis
        influence_metrics = self._analyze_user_influence(user_id)
        
        return {
            "similar_users": similar_users[:5],  # Top 5 for display
            "user_cluster": user_cluster,
            "influence_metrics": influence_metrics,
            "community_position": self._determine_community_position(user_id)
        }
    
    def _create_predictive_profile(self, user_id: str) -> Dict[str, Any]:
        """Create predictive insights about user's future donation behavior."""
        user_donations = self.donations.get(user_id, [])
        
        # Predict next donation amount
        predicted_amount = self._predict_next_donation_amount(user_donations)
        
        # Predict category interests
        predicted_categories = self._predict_category_interests(user_id)
        
        # Predict donation likelihood
        donation_likelihood = self._calculate_donation_likelihood(user_donations)
        
        return {
            "predicted_next_amount": predicted_amount,
            "predicted_categories": predicted_categories,
            "donation_likelihood": donation_likelihood,
            "growth_potential": self._assess_growth_potential(user_donations),
            "churn_risk": self._assess_churn_risk(user_donations)
        }
    
    def _analyze_donation_consistency(self, amounts: List[float]) -> Dict[str, Any]:
        """Analyze how consistent user's donation amounts are."""
        if len(amounts) < 2:
            return {"consistency_score": 1.0, "pattern": "insufficient_data"}
        
        mean_amount = statistics.mean(amounts)
        std_dev = statistics.stdev(amounts)
        cv = std_dev / mean_amount if mean_amount > 0 else 0  # Coefficient of variation
        
        # Consistency score (lower CV = higher consistency)
        consistency_score = 1 / (1 + cv)
        
        # Pattern classification
        if cv < 0.3:
            pattern = "highly_consistent"
        elif cv < 0.7:
            pattern = "moderately_consistent"
        else:
            pattern = "highly_variable"
        
        return {
            "consistency_score": consistency_score,
            "coefficient_of_variation": cv,
            "pattern": pattern,
            "amount_trend": self._detect_amount_trend(amounts)
        }
    
    def _detect_amount_trend(self, amounts: List[float]) -> str:
        """Detect if donation amounts are increasing, decreasing, or stable."""
        if len(amounts) < 3:
            return "insufficient_data"
        
        # Simple trend analysis using first and last third
        first_third = amounts[:len(amounts)//3]
        last_third = amounts[-len(amounts)//3:]
        
        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)
        
        change_ratio = (last_avg - first_avg) / first_avg if first_avg > 0 else 0
        
        if change_ratio > 0.2:
            return "increasing"
        elif change_ratio < -0.2:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_engagement_level(self, user_donations: List[Dict]) -> Dict[str, Any]:
        """Calculate user's engagement level with the platform."""
        total_donations = len(user_donations)
        unique_campaigns = len(set(d['campaign_id'] for d in user_donations))
        
        # Engagement metrics
        campaign_diversity = unique_campaigns / total_donations if total_donations > 0 else 0
        repeat_donation_rate = (total_donations - unique_campaigns) / total_donations if total_donations > 0 else 0
        
        # Engagement level classification
        if total_donations >= 10:
            level = "high"
        elif total_donations >= 5:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "total_donations": total_donations,
            "campaign_diversity": campaign_diversity,
            "repeat_donation_rate": repeat_donation_rate,
            "engagement_score": min(total_donations / 10, 1.0)
        }
    
    def _analyze_risk_tolerance(self, user_id: str, user_donations: List[Dict]) -> Dict[str, Any]:
        """Analyze user's risk tolerance based on campaign choices."""
        risk_scores = []
        
        for donation in user_donations:
            campaign_id = str(donation['campaign_id'])
            campaign = self.campaign_lookup.get(campaign_id, {})
            
            # Risk factors
            target_amount = campaign.get('target_amount', 0)
            is_active = campaign.get('is_active', True)
            
            # Higher target = higher risk (less likely to succeed)
            target_risk = min(target_amount / 100000000, 1.0)  # Normalize to 100M
            
            # Inactive campaigns = higher risk
            status_risk = 0.0 if is_active else 0.5
            
            total_risk = (target_risk + status_risk) / 2
            risk_scores.append(total_risk)
        
        avg_risk = statistics.mean(risk_scores) if risk_scores else 0
        
        if avg_risk < 0.3:
            tolerance = "conservative"
        elif avg_risk < 0.6:
            tolerance = "moderate"
        else:
            tolerance = "aggressive"
        
        return {
            "tolerance_level": tolerance,
            "average_risk_score": avg_risk,
            "risk_distribution": self._categorize_risk_scores(risk_scores)
        }
    
    def _categorize_risk_scores(self, risk_scores: List[float]) -> Dict[str, int]:
        """Categorize risk scores into buckets."""
        categories = {"low": 0, "medium": 0, "high": 0}
        
        for score in risk_scores:
            if score < 0.3:
                categories["low"] += 1
            elif score < 0.6:
                categories["medium"] += 1
            else:
                categories["high"] += 1
        
        return categories
    
    def _calculate_donation_momentum(self, user_donations: List[Dict]) -> Dict[str, Any]:
        """Calculate user's donation momentum (recent activity vs historical)."""
        if len(user_donations) < 2:
            return {"momentum": "insufficient_data"}
        
        # For simplicity, assume donations are ordered (in real scenario, use timestamps)
        total_donations = len(user_donations)
        recent_donations = user_donations[-max(1, total_donations//3):]  # Last third
        
        recent_count = len(recent_donations)
        recent_avg_amount = statistics.mean([d['amount'] for d in recent_donations])
        overall_avg_amount = statistics.mean([d['amount'] for d in user_donations])
        
        # Momentum indicators
        frequency_momentum = recent_count / (total_donations / 3) if total_donations > 3 else 1
        amount_momentum = recent_avg_amount / overall_avg_amount if overall_avg_amount > 0 else 1
        
        overall_momentum = (frequency_momentum + amount_momentum) / 2
        
        if overall_momentum > 1.2:
            momentum_type = "accelerating"
        elif overall_momentum > 0.8:
            momentum_type = "stable"
        else:
            momentum_type = "declining"
        
        return {
            "momentum": momentum_type,
            "momentum_score": overall_momentum,
            "frequency_momentum": frequency_momentum,
            "amount_momentum": amount_momentum
        }
    
    def _classify_behavioral_type(self, consistency_metrics: Dict, engagement_level: Dict) -> str:
        """Classify user into behavioral types."""
        consistency = consistency_metrics.get("consistency_score", 0)
        engagement = engagement_level.get("engagement_score", 0)
        
        if engagement > 0.7 and consistency > 0.7:
            return "loyal_consistent"
        elif engagement > 0.7 and consistency <= 0.7:
            return "loyal_variable"
        elif engagement <= 0.7 and consistency > 0.7:
            return "occasional_consistent"
        else:
            return "occasional_variable"
    
    def _analyze_category_semantics(self, user_id: str) -> Dict[str, Any]:
        """Analyze category preferences with semantic understanding."""
        user_donations = self.donations.get(user_id, [])
        
        # Group categories by semantic similarity
        category_groups = {
            "health_medical": ["Health", "Medical", "Healthcare"],
            "education_development": ["Education", "Development", "Training"],
            "social_welfare": ["Social", "Welfare", "Community"],
            "emergency_disaster": ["Emergency", "Disaster", "Relief"],
            "religious_spiritual": ["Religious", "Spiritual", "Faith"]
        }
        
        group_preferences = defaultdict(list)
        
        for donation in user_donations:
            campaign_id = str(donation['campaign_id'])
            campaign = self.campaign_lookup.get(campaign_id, {})
            category_name = campaign.get('category_name', '').lower()
            
            # Map to semantic groups
            for group, keywords in category_groups.items():
                if any(keyword.lower() in category_name for keyword in keywords):
                    group_preferences[group].append(donation['amount'])
                    break
            else:
                group_preferences['other'].append(donation['amount'])
        
        # Calculate group statistics
        group_stats = {}
        for group, amounts in group_preferences.items():
            if amounts:
                group_stats[group] = {
                    "donation_count": len(amounts),
                    "total_amount": sum(amounts),
                    "avg_amount": statistics.mean(amounts),
                    "preference_strength": len(amounts) / len(user_donations)
                }
        
        return {
            "semantic_groups": group_stats,
            "dominant_theme": max(group_stats.keys(), key=lambda k: group_stats[k]["preference_strength"]) if group_stats else None
        }
    
    def _analyze_geographic_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze geographic donation preferences."""
        user_donations = self.donations.get(user_id, [])
        
        country_stats = defaultdict(lambda: {"count": 0, "amount": 0})
        
        for donation in user_donations:
            campaign_id = str(donation['campaign_id'])
            campaign = self.campaign_lookup.get(campaign_id, {})
            country = campaign.get('country', 'Unknown')
            
            country_stats[country]["count"] += 1
            country_stats[country]["amount"] += donation['amount']
        
        # Calculate preferences
        total_donations = len(user_donations)
        total_amount = sum(d['amount'] for d in user_donations)
        
        geographic_preferences = {}
        for country, stats in country_stats.items():
            geographic_preferences[country] = {
                "donation_count": stats["count"],
                "total_amount": stats["amount"],
                "frequency_preference": stats["count"] / total_donations,
                "amount_preference": stats["amount"] / total_amount
            }
        
        return {
            "country_preferences": geographic_preferences,
            "geographic_diversity": len(country_stats),
            "primary_region": max(geographic_preferences.keys(), 
                                key=lambda k: geographic_preferences[k]["frequency_preference"]) if geographic_preferences else None
        }
    
    def _analyze_campaign_size_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze preferences for campaign sizes."""
        user_donations = self.donations.get(user_id, [])
        
        size_categories = {"small": 0, "medium": 0, "large": 0, "mega": 0}
        
        for donation in user_donations:
            campaign_id = str(donation['campaign_id'])
            campaign = self.campaign_lookup.get(campaign_id, {})
            target = campaign.get('target_amount', 0)
            
            if target < 10000000:  # < 10M
                size_categories["small"] += 1
            elif target < 50000000:  # < 50M
                size_categories["medium"] += 1
            elif target < 200000000:  # < 200M
                size_categories["large"] += 1
            else:
                size_categories["mega"] += 1
        
        total = sum(size_categories.values())
        size_preferences = {k: v/total for k, v in size_categories.items()} if total > 0 else size_categories
        
        return {
            "size_distribution": size_categories,
            "size_preferences": size_preferences,
            "preferred_size": max(size_preferences.keys(), key=lambda k: size_preferences[k]) if any(size_preferences.values()) else None
        }
    
    def _analyze_impact_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's preference for donation impact level."""
        user_donations = self.donations.get(user_id, [])
        
        impact_scores = []
        
        for donation in user_donations:
            campaign_id = str(donation['campaign_id'])
            campaign = self.campaign_lookup.get(campaign_id, {})
            target = campaign.get('target_amount', 1)
            
            # Impact = donation amount / campaign target
            impact = donation['amount'] / target if target > 0 else 0
            impact_scores.append(impact)
        
        if not impact_scores:
            return {"error": "No impact data available"}
        
        avg_impact = statistics.mean(impact_scores)
        
        if avg_impact > 0.01:  # > 1% impact
            impact_preference = "high_impact"
        elif avg_impact > 0.001:  # > 0.1% impact
            impact_preference = "medium_impact"
        else:
            impact_preference = "low_impact"
        
        return {
            "average_impact": avg_impact,
            "impact_preference": impact_preference,
            "impact_distribution": self._categorize_impact_scores(impact_scores)
        }
    
    def _categorize_impact_scores(self, impact_scores: List[float]) -> Dict[str, int]:
        """Categorize impact scores."""
        categories = {"high": 0, "medium": 0, "low": 0}
        
        for score in impact_scores:
            if score > 0.01:
                categories["high"] += 1
            elif score > 0.001:
                categories["medium"] += 1
            else:
                categories["low"] += 1
        
        return categories
    
    def _calculate_preference_strength(self, user_donations: List[Dict]) -> float:
        """Calculate overall strength of user preferences."""
        if len(user_donations) < 2:
            return 0.0
        
        # Factors that indicate strong preferences
        unique_campaigns = len(set(d['campaign_id'] for d in user_donations))
        repeat_rate = (len(user_donations) - unique_campaigns) / len(user_donations)
        
        # Strong preferences indicated by repeat donations and consistency
        preference_strength = min(repeat_rate * 2, 1.0)  # Cap at 1.0
        
        return preference_strength
    
    def _identify_user_cluster(self, user_id: str, similar_users: List[Dict]) -> Dict[str, Any]:
        """Identify which cluster/segment the user belongs to."""
        if not similar_users:
            return {"cluster": "unique", "confidence": 0.0}
        
        # Simple clustering based on similarity scores
        top_similarity = similar_users[0]["similarity_score"] if similar_users else 0
        
        if top_similarity > 0.8:
            cluster = "highly_similar_group"
        elif top_similarity > 0.6:
            cluster = "moderately_similar_group"
        elif top_similarity > 0.4:
            cluster = "loosely_similar_group"
        else:
            cluster = "unique_donor"
        
        return {
            "cluster": cluster,
            "confidence": top_similarity,
            "cluster_size": len([u for u in similar_users if u["similarity_score"] > 0.5])
        }
    
    def _analyze_user_influence(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's potential influence in the donor community."""
        user_profile = self.analyzer.get_user_donation_profile(user_id)
        
        if "error" in user_profile:
            return {"influence": "unknown"}
        
        # Influence factors
        total_amount = user_profile["basic_stats"]["total_amount"]
        total_donations = user_profile["basic_stats"]["total_donations"]
        generosity_score = user_profile["generosity_score"]
        diversity_score = user_profile["diversity_score"]
        
        # Calculate influence score
        amount_influence = min(total_amount / 5000000, 1.0)  # Max at 5M
        frequency_influence = min(total_donations / 20, 1.0)  # Max at 20 donations
        
        overall_influence = (amount_influence * 0.4 + frequency_influence * 0.3 + 
                           generosity_score * 0.2 + diversity_score * 0.1)
        
        if overall_influence > 0.8:
            influence_level = "high"
        elif overall_influence > 0.5:
            influence_level = "medium"
        else:
            influence_level = "low"
        
        return {
            "influence_level": influence_level,
            "influence_score": overall_influence,
            "amount_influence": amount_influence,
            "frequency_influence": frequency_influence
        }
    
    def _determine_community_position(self, user_id: str) -> str:
        """Determine user's position in the donor community."""
        user_profile = self.analyzer.get_user_donation_profile(user_id)
        
        if "error" in user_profile:
            return "unknown"
        
        generosity = user_profile["generosity_score"]
        diversity = user_profile["diversity_score"]
        
        if generosity > 0.7 and diversity > 0.7:
            return "community_leader"
        elif generosity > 0.7:
            return "generous_specialist"
        elif diversity > 0.7:
            return "diverse_supporter"
        elif generosity > 0.4 or diversity > 0.4:
            return "active_member"
        else:
            return "casual_donor"
    
    def _predict_next_donation_amount(self, user_donations: List[Dict]) -> Dict[str, Any]:
        """Predict user's next donation amount."""
        if len(user_donations) < 2:
            return {"prediction": "insufficient_data"}
        
        amounts = [d['amount'] for d in user_donations]
        
        # Simple prediction based on recent trend and average
        recent_amounts = amounts[-3:] if len(amounts) >= 3 else amounts
        trend_factor = self._calculate_trend_factor(amounts)
        
        base_prediction = statistics.mean(recent_amounts)
        adjusted_prediction = base_prediction * trend_factor
        
        # Confidence based on consistency
        std_dev = statistics.stdev(amounts) if len(amounts) > 1 else 0
        mean_amount = statistics.mean(amounts)
        confidence = 1 / (1 + std_dev / mean_amount) if mean_amount > 0 else 0
        
        return {
            "predicted_amount": adjusted_prediction,
            "confidence": confidence,
            "range_low": adjusted_prediction * 0.7,
            "range_high": adjusted_prediction * 1.3,
            "trend_factor": trend_factor
        }
    
    def _calculate_trend_factor(self, amounts: List[float]) -> float:
        """Calculate trend factor for amount prediction."""
        if len(amounts) < 3:
            return 1.0
        
        # Compare recent vs earlier amounts
        recent = amounts[-len(amounts)//3:]
        earlier = amounts[:len(amounts)//3]
        
        recent_avg = statistics.mean(recent)
        earlier_avg = statistics.mean(earlier)
        
        if earlier_avg > 0:
            trend_factor = recent_avg / earlier_avg
            return max(0.5, min(2.0, trend_factor))  # Cap between 0.5 and 2.0
        
        return 1.0
    
    def _predict_category_interests(self, user_id: str) -> Dict[str, Any]:
        """Predict user's future category interests."""
        user_profile = self.analyzer.get_user_donation_profile(user_id)
        
        if "error" in user_profile:
            return {"prediction": "insufficient_data"}
        
        current_preferences = user_profile["category_preferences"]["preferences"]
        
        # Predict based on current strong preferences and similar users
        similar_users = self.analyzer.find_similar_users(user_id, top_k=5)
        
        predicted_categories = {}
        
        # Weight current preferences
        for category, stats in current_preferences.items():
            predicted_categories[category] = stats["combined_score"] * 0.8
        
        # Add categories from similar users
        for similar_user in similar_users:
            similar_profile = self.analyzer.get_user_donation_profile(similar_user["user_id"])
            if "error" not in similar_profile:
                similar_prefs = similar_profile["category_preferences"]["preferences"]
                for category, stats in similar_prefs.items():
                    if category not in predicted_categories:
                        predicted_categories[category] = stats["combined_score"] * 0.3 * similar_user["similarity_score"]
        
        # Sort by predicted interest
        sorted_predictions = dict(sorted(
            predicted_categories.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return {
            "predicted_interests": sorted_predictions,
            "new_category_suggestions": [cat for cat in sorted_predictions.keys() 
                                       if cat not in current_preferences][:3]
        }
    
    def _calculate_donation_likelihood(self, user_donations: List[Dict]) -> Dict[str, Any]:
        """Calculate likelihood of user making another donation."""
        if not user_donations:
            return {"likelihood": "unknown"}
        
        # Factors affecting likelihood
        recent_activity = len(user_donations) >= 3  # Has recent activity
        consistency = len(user_donations) > 1  # Has donated more than once
        momentum = self._calculate_donation_momentum(user_donations)
        
        # Base likelihood
        base_likelihood = 0.5
        
        # Adjust based on factors
        if recent_activity:
            base_likelihood += 0.2
        if consistency:
            base_likelihood += 0.2
        if momentum.get("momentum") == "accelerating":
            base_likelihood += 0.2
        elif momentum.get("momentum") == "declining":
            base_likelihood -= 0.2
        
        likelihood_score = max(0.1, min(0.9, base_likelihood))
        
        if likelihood_score > 0.7:
            likelihood_level = "high"
        elif likelihood_score > 0.4:
            likelihood_level = "medium"
        else:
            likelihood_level = "low"
        
        return {
            "likelihood_level": likelihood_level,
            "likelihood_score": likelihood_score,
            "factors": {
                "recent_activity": recent_activity,
                "consistency": consistency,
                "momentum": momentum.get("momentum", "unknown")
            }
        }
    
    def _assess_growth_potential(self, user_donations: List[Dict]) -> Dict[str, Any]:
        """Assess user's potential for donation growth."""
        if len(user_donations) < 2:
            return {"potential": "unknown"}
        
        amounts = [d['amount'] for d in user_donations]
        
        # Growth indicators
        trend = self._detect_amount_trend(amounts)
        consistency = statistics.stdev(amounts) / statistics.mean(amounts) if statistics.mean(amounts) > 0 else 0
        frequency = len(user_donations)
        
        # Calculate growth potential
        growth_score = 0.5  # Base score
        
        if trend == "increasing":
            growth_score += 0.3
        elif trend == "stable":
            growth_score += 0.1
        
        if consistency < 0.5:  # Low variability = more predictable growth
            growth_score += 0.2
        
        if frequency > 5:  # High frequency = more engaged
            growth_score += 0.2
        
        growth_score = max(0.1, min(1.0, growth_score))
        
        if growth_score > 0.7:
            potential_level = "high"
        elif growth_score > 0.4:
            potential_level = "medium"
        else:
            potential_level = "low"
        
        return {
            "potential_level": potential_level,
            "growth_score": growth_score,
            "trend": trend,
            "consistency": consistency
        }
    
    def _assess_churn_risk(self, user_donations: List[Dict]) -> Dict[str, Any]:
        """Assess risk of user stopping donations."""
        if not user_donations:
            return {"risk": "unknown"}
        
        # Risk factors
        donation_count = len(user_donations)
        momentum = self._calculate_donation_momentum(user_donations)
        
        # Calculate churn risk
        risk_score = 0.5  # Base risk
        
        if donation_count == 1:
            risk_score += 0.3  # One-time donors have higher churn risk
        elif donation_count < 3:
            risk_score += 0.2
        else:
            risk_score -= 0.2  # Repeat donors have lower churn risk
        
        if momentum.get("momentum") == "declining":
            risk_score += 0.3
        elif momentum.get("momentum") == "accelerating":
            risk_score -= 0.3
        
        risk_score = max(0.1, min(0.9, risk_score))
        
        if risk_score > 0.6:
            risk_level = "high"
        elif risk_score > 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "churn_score": risk_score,
            "factors": {
                "donation_count": donation_count,
                "momentum": momentum.get("momentum", "unknown")
            }
        }
    
    def _calculate_profile_completeness(self, base_profile: Dict[str, Any]) -> float:
        """Calculate how complete the user profile is."""
        completeness_factors = [
            base_profile["basic_stats"]["total_donations"] > 0,
            len(base_profile["category_preferences"]["preferences"]) > 0,
            base_profile["basic_stats"]["total_donations"] > 3,
            base_profile["diversity_score"] > 0.3,
            base_profile["generosity_score"] > 0.3
        ]
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _calculate_recommendation_confidence(self, base_profile: Dict[str, Any]) -> float:
        """Calculate confidence level for recommendations."""
        confidence_factors = [
            base_profile["basic_stats"]["total_donations"] / 10,  # More donations = higher confidence
            base_profile["diversity_score"],  # More diverse = better understanding
            len(base_profile["category_preferences"]["preferences"]) / 5,  # More categories = better profiling
            min(base_profile["basic_stats"]["unique_campaigns"] / 5, 1.0)  # More campaigns = better data
        ]
        
        return min(sum(confidence_factors) / len(confidence_factors), 1.0)


def example_usage():
    """Demonstrate the advanced profiling capabilities."""
    profiler = AdvancedUserProfiler()
    
    print("=== ADVANCED USER PROFILING DEMO ===\n")
    
    user_id = "1001"
    print(f"🧠 Advanced Profile for User {user_id}:")
    print("=" * 60)
    
    profile = profiler.create_comprehensive_profile(user_id)
    
    if "error" in profile:
        print(f"Error: {profile['error']}")
        return
    
    # Profile completeness and confidence
    print(f"📊 Profile Completeness: {profile['profile_completeness']:.1%}")
    print(f"🎯 Recommendation Confidence: {profile['recommendation_confidence']:.1%}")
    
    # Behavioral profile
    behavioral = profile["behavioral_profile"]
    print(f"\n🎭 Behavioral Profile:")
    print(f"  • Type: {behavioral['behavioral_type']}")
    print(f"  • Engagement Level: {behavioral['engagement_level']['level']}")
    print(f"  • Risk Tolerance: {behavioral['risk_tolerance']['tolerance_level']}")
    print(f"  • Donation Momentum: {behavioral['donation_momentum']['momentum']}")
    
    # Preference profile
    preference = profile["preference_profile"]
    print(f"\n💡 Preference Insights:")
    if "semantic_groups" in preference["category_insights"]:
        dominant = preference["category_insights"]["dominant_theme"]
        print(f"  • Dominant Theme: {dominant}")
    
    geographic = preference["geographic_preferences"]
    if geographic["primary_region"]:
        print(f"  • Primary Region: {geographic['primary_region']}")
    
    size_pref = preference["size_preferences"]["preferred_size"]
    print(f"  • Preferred Campaign Size: {size_pref}")
    
    # Social profile
    social = profile["social_profile"]
    print(f"\n👥 Social Profile:")
    print(f"  • Community Position: {social['community_position']}")
    print(f"  • User Cluster: {social['user_cluster']['cluster']}")
    print(f"  • Influence Level: {social['influence_metrics']['influence_level']}")
    
    # Predictive insights
    predictive = profile["predictive_profile"]
    print(f"\n🔮 Predictive Insights:")
    
    next_amount = predictive["predicted_next_amount"]
    if "predicted_amount" in next_amount:
        print(f"  • Next Donation (predicted): IDR {next_amount['predicted_amount']:,.0f}")
        print(f"    Range: IDR {next_amount['range_low']:,.0f} - {next_amount['range_high']:,.0f}")
        print(f"    Confidence: {next_amount['confidence']:.1%}")
    
    likelihood = predictive["donation_likelihood"]
    print(f"  • Donation Likelihood: {likelihood['likelihood_level']} ({likelihood['likelihood_score']:.1%})")
    
    growth = predictive["growth_potential"]
    print(f"  • Growth Potential: {growth['potential_level']}")
    
    churn = predictive["churn_risk"]
    print(f"  • Churn Risk: {churn['risk_level']}")
    
    # New category suggestions
    if "predicted_category_interests" in predictive:
        category_pred = predictive["predicted_category_interests"]
        if "new_category_suggestions" in category_pred:
            new_cats = category_pred["new_category_suggestions"]
            if new_cats:
                print(f"\n🆕 New Category Suggestions:")
                for i, cat in enumerate(new_cats, 1):
                    print(f"  {i}. {cat}")


if __name__ == "__main__":
    example_usage()