"""
Advanced User Profiling System - Database Version

This module creates sophisticated user profiles that combine multiple signals
from donation history to enable highly personalized recommendations.
Uses database instead of JSON files for data retrieval.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter
import statistics
import os
from .donation_analyzer_db import DatabaseDonationAnalyzer
from .database import DatabaseManager
from .models import ProjectRepository, DonationRepository, UserRepository
from .user_prefs_store import UserPreferencesStore


class AdvancedUserProfilerDB:
    """Creates comprehensive user profiles for personalized recommendations using database."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the profiler with database manager."""
        self.db_manager = db_manager
        # Use DatabaseDonationAnalyzer (auto init DB via env)
        self.analyzer = DatabaseDonationAnalyzer(auto_init_db=True)
        self.project_repo = ProjectRepository(db_manager)
        self.donation_repo = DonationRepository(db_manager)
        self.user_repo = UserRepository(db_manager)
        # Optional JSON-based user preferences store
        self.use_json_prefs = os.getenv('USER_PREFERENCES_JSON', '0') == '1'
        prefs_path = os.getenv('USER_PREFERENCES_JSON_PATH', 'data/user_preferences.json')
        self.prefs_store = UserPreferencesStore(prefs_path) if self.use_json_prefs else None
    
    def create_comprehensive_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Create a comprehensive user profile that combines multiple dimensions:
        - Behavioral patterns (frequency, consistency, timing)
        - Preference patterns (categories, amounts, geography)
        - Social patterns (similarity to other users)
        - Predictive patterns (likely future behavior)
        """

        print(f'calling create_comprehensive_profile for user_id: {user_id}')
        base_profile = self.analyzer.get_user_donation_profile(user_id)
        
        if "error" in base_profile:
            return base_profile
        
        # Add advanced profiling layers
        behavioral_profile = self._create_behavioral_profile(user_id)
        preference_profile = self._create_preference_profile(user_id)
        # Persist preference profile to JSON if configured
        if self.prefs_store:
            try:
                self.prefs_store.set_user_preferences(user_id, preference_profile)
            except Exception:
                # Non-fatal: preferences persistence should not break profiling
                pass
        social_profile = self._create_social_profile(user_id)
        predictive_profile = self._create_predictive_profile(user_id)
        
        return {
            **base_profile,
            "behavioral_profile": behavioral_profile,
            "preference_profile": preference_profile,
            "social_profile": social_profile,
            "predictive_profile": predictive_profile,
            "profile_metadata": {
                "completeness_score": self._calculate_profile_completeness(base_profile),
                "confidence_score": self._calculate_recommendation_confidence(base_profile),
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def _create_behavioral_profile(self, user_id: str) -> Dict[str, Any]:
        """Analyze behavioral patterns in donation history."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        if not user_donations:
            return {"error": "No donations found for behavioral analysis"}
        
        amounts = [d.amount for d in user_donations]
        
        # Analyze donation consistency
        consistency_metrics = self._analyze_donation_consistency(amounts)
        
        # Calculate engagement level
        engagement_level = self._calculate_engagement_level(user_donations)
        
        # Analyze risk tolerance
        risk_analysis = self._analyze_risk_tolerance(user_id, user_donations)
        
        # Calculate donation momentum
        momentum = self._calculate_donation_momentum(user_donations)
        
        # Classify behavioral type
        behavioral_type = self._classify_behavioral_type(consistency_metrics, engagement_level)
        
        return {
            "consistency_metrics": consistency_metrics,
            "engagement_level": engagement_level,
            "risk_tolerance": risk_analysis,
            "donation_momentum": momentum,
            "behavioral_type": behavioral_type
        }
    
    def _create_preference_profile(self, user_id: str) -> Dict[str, Any]:
        """Analyze preference patterns in donation choices."""
        # Category semantic analysis
        category_analysis = self._analyze_category_semantics(user_id)
        
        # Geographic preferences
        geographic_prefs = self._analyze_geographic_preferences(user_id)
        
        # Campaign size preferences
        size_prefs = self._analyze_campaign_size_preferences(user_id)
        
        # Impact preferences
        impact_prefs = self._analyze_impact_preferences(user_id)
        
        return {
            "category_semantics": category_analysis,
            "geographic_preferences": geographic_prefs,
            "campaign_size_preferences": size_prefs,
            "impact_preferences": impact_prefs
        }
    
    def _create_social_profile(self, user_id: str) -> Dict[str, Any]:
        """Analyze social patterns and user clustering."""
        # Find similar users
        similar_users = self.analyzer.find_similar_users(user_id, top_k=10)
        
        # Identify user cluster
        cluster_info = self._identify_user_cluster(user_id, similar_users)
        
        # Analyze user influence
        influence_metrics = self._analyze_user_influence(user_id)
        
        return {
            "similar_users": similar_users,
            "cluster_information": cluster_info,
            "influence_metrics": influence_metrics
        }
    
    def _create_predictive_profile(self, user_id: str) -> Dict[str, Any]:
        """Create predictive insights about future donation behavior."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        # Predict next donation amount
        amount_prediction = self._predict_next_donation_amount(user_donations)
        
        # Predict category interests
        category_prediction = self._predict_category_interests(user_id)
        
        # Calculate donation likelihood
        likelihood_analysis = self._calculate_donation_likelihood(user_donations)
        
        # Assess growth potential
        growth_potential = self._assess_growth_potential(user_donations)
        
        # Assess churn risk
        churn_risk = self._assess_churn_risk(user_donations)
        
        return {
            "next_donation_prediction": amount_prediction,
            "category_interest_prediction": category_prediction,
            "donation_likelihood": likelihood_analysis,
            "growth_potential": growth_potential,
            "churn_risk": churn_risk
        }
    
    def _analyze_donation_consistency(self, amounts: List[float]) -> Dict[str, Any]:
        """Analyze consistency patterns in donation amounts."""
        if len(amounts) < 2:
            return {"consistency_score": 0, "pattern": "insufficient_data"}
        
        # Calculate coefficient of variation
        mean_amount = statistics.mean(amounts)
        std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
        cv = std_amount / mean_amount if mean_amount > 0 else 0
        
        # Consistency score (inverse of coefficient of variation)
        consistency_score = max(0, 1 - cv)
        
        # Detect trend
        trend = self._detect_amount_trend(amounts)
        
        # Calculate donation frequency consistency
        frequency_consistency = self._calculate_frequency_consistency(amounts)
        
        return {
            "consistency_score": consistency_score,
            "coefficient_of_variation": cv,
            "amount_trend": trend,
            "frequency_consistency": frequency_consistency,
            "pattern": "consistent" if consistency_score > 0.7 else "variable"
        }
    
    def _detect_amount_trend(self, amounts: List[float]) -> str:
        """Detect trend in donation amounts over time."""
        if len(amounts) < 3:
            return "insufficient_data"
        
        # Simple linear regression to detect trend
        x = list(range(len(amounts)))
        y = amounts
        
        n = len(amounts)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_frequency_consistency(self, amounts: List[float]) -> float:
        """Calculate how consistent the donation frequency is."""
        # This is a simplified version - in practice, you'd use actual timestamps
        return 0.8  # Placeholder
    
    def _calculate_engagement_level(self, user_donations: List) -> Dict[str, Any]:
        """Calculate user engagement level based on donation patterns."""
        if not user_donations:
            return {"level": "none", "score": 0}
        
        # Calculate various engagement metrics
        total_donations = len(user_donations)
        total_amount = sum(d.amount for d in user_donations)
        avg_amount = total_amount / total_donations
        
        # Engagement score based on frequency and amount
        frequency_score = min(1.0, total_donations / 10)  # Normalize to max 10 donations
        amount_score = min(1.0, avg_amount / 1000000)  # Normalize to max 1M IDR
        
        engagement_score = (frequency_score + amount_score) / 2
        
        if engagement_score >= 0.8:
            level = "high"
        elif engagement_score >= 0.5:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": engagement_score,
            "frequency_score": frequency_score,
            "amount_score": amount_score,
            "total_donations": total_donations,
            "average_amount": avg_amount
        }
    
    def _analyze_risk_tolerance(self, user_id: str, user_donations: List) -> Dict[str, Any]:
        """Analyze user's risk tolerance based on campaign choices."""
        if not user_donations:
            return {"tolerance": "unknown", "score": 0.5}
        
        risk_scores = []
        for donation in user_donations:
            project = self.project_repo.get_by_id(donation.project_id)
            if project:
                # Calculate risk score based on project characteristics
                risk_score = self._calculate_campaign_risk_score(project)
                risk_scores.append(risk_score)
        
        if not risk_scores:
            return {"tolerance": "unknown", "score": 0.5}
        
        avg_risk = statistics.mean(risk_scores)
        risk_distribution = self._categorize_risk_scores(risk_scores)
        
        # Determine risk tolerance level
        if avg_risk >= 0.7:
            tolerance = "high"
        elif avg_risk >= 0.4:
            tolerance = "medium"
        else:
            tolerance = "low"
        
        return {
            "tolerance": tolerance,
            "average_risk_score": avg_risk,
            "risk_distribution": risk_distribution,
            "consistency": statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0
        }
    
    def _calculate_campaign_risk_score(self, campaign) -> float:
        """Calculate risk score for a campaign."""
        risk_score = 0.5  # Base score
        
        # Adjust based on campaign characteristics
        if hasattr(campaign, 'target_amount') and campaign.target_amount:
            if campaign.target_amount > 100000000:  # 100M IDR
                risk_score += 0.2
        
        high_risk_categories = ['emergency', 'disaster', 'medical']
        # Category may be a name or a numeric foreign key; handle gracefully
        if hasattr(campaign, 'category') and campaign.category is not None:
            cat_val = campaign.category
            cat_name = None
            if isinstance(cat_val, str):
                cat_name = cat_val.lower()
            elif isinstance(cat_val, (int, float)):
                cat_name = None  # Numeric ID without lookup; skip risk by category
            else:
                try:
                    cat_name = str(cat_val).lower()
                except Exception:
                    cat_name = None
            if cat_name and cat_name in high_risk_categories:
                risk_score += 0.3
        
        # Fallback: consider project_type and urgency_level if present
        if hasattr(campaign, 'project_type') and campaign.project_type:
            try:
                pt = str(campaign.project_type).lower()
                if any(k in pt for k in high_risk_categories):
                    risk_score += 0.2
            except Exception:
                pass
        if hasattr(campaign, 'urgency_level') and campaign.urgency_level:
            try:
                ul = str(campaign.urgency_level).lower()
                if any(k in ul for k in ['emergency', 'critical']):
                    risk_score += 0.2
            except Exception:
                pass
        
        return min(1.0, risk_score)
    
    def _categorize_risk_scores(self, risk_scores: List[float]) -> Dict[str, int]:
        """Categorize risk scores into low, medium, high."""
        categories = {"low": 0, "medium": 0, "high": 0}
        
        for score in risk_scores:
            if score < 0.4:
                categories["low"] += 1
            elif score < 0.7:
                categories["medium"] += 1
            else:
                categories["high"] += 1
        
        return categories
    
    def _calculate_donation_momentum(self, user_donations: List) -> Dict[str, Any]:
        """Calculate donation momentum and trends."""
        if len(user_donations) < 2:
            return {"momentum": "insufficient_data", "trend": "unknown"}
        
        # Sort by date (normalize to datetime to avoid None/str issues)
        def _to_datetime(donation) -> datetime:
            # Try created_at, then donation_date
            raw = getattr(donation, 'created_at', None) or getattr(donation, 'donation_date', None)
            if not raw:
                return datetime.min
            try:
                # Prefer ISO 8601 if available
                return datetime.fromisoformat(str(raw))
            except Exception:
                # Try common formats
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                    try:
                        return datetime.strptime(str(raw), fmt)
                    except Exception:
                        continue
                # Fallback to minimal datetime to keep ordering consistent
                return datetime.min
        sorted_donations = sorted(user_donations, key=_to_datetime)
        
        # Calculate momentum based on recent activity
        recent_donations = sorted_donations[-3:] if len(sorted_donations) >= 3 else sorted_donations
        total_recent = sum(d.amount for d in recent_donations)
        
        # Compare with historical average
        historical_avg = sum(d.amount for d in user_donations) / len(user_donations)
        recent_avg = total_recent / len(recent_donations)
        
        momentum_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
        
        if momentum_ratio > 1.2:
            momentum = "increasing"
        elif momentum_ratio < 0.8:
            momentum = "decreasing"
        else:
            momentum = "stable"
        
        return {
            "momentum": momentum,
            "momentum_ratio": momentum_ratio,
            "recent_average": recent_avg,
            "historical_average": historical_avg,
            "trend_strength": abs(momentum_ratio - 1)
        }
    
    def _classify_behavioral_type(self, consistency_metrics: Dict, engagement_level: Dict) -> str:
        """Classify user into behavioral types."""
        consistency_score = consistency_metrics.get("consistency_score", 0)
        engagement_score = engagement_level.get("score", 0)
        
        if engagement_score >= 0.8 and consistency_score >= 0.7:
            return "champion"
        elif engagement_score >= 0.6:
            return "supporter"
        elif consistency_score >= 0.7:
            return "steady_giver"
        elif engagement_score >= 0.3:
            return "occasional_donor"
        else:
            return "new_donor"
    
    # Additional methods for preference analysis
    def _analyze_category_semantics(self, user_id: str) -> Dict[str, Any]:
        """Analyze semantic patterns in category preferences."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        category_counts = defaultdict(int)
        for donation in user_donations:
            project = self.project_repo.get_by_id(donation.project_id)
            if project and hasattr(project, 'category'):
                category_counts[project.category] += 1
        
        total_donations = len(user_donations)
        category_preferences = {
            cat: count / total_donations 
            for cat, count in category_counts.items()
        } if total_donations > 0 else {}
        
        return {
            "category_distribution": dict(category_counts),
            "category_preferences": category_preferences,
            "primary_categories": sorted(category_preferences.items(), 
                                       key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _analyze_geographic_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze geographic donation patterns."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        location_counts = defaultdict(int)
        for donation in user_donations:
            project = self.project_repo.get_by_id(donation.project_id)
            if project and hasattr(project, 'location'):
                location_counts[project.location] += 1
        
        total_donations = len(user_donations)
        location_preferences = {
            loc: count / total_donations 
            for loc, count in location_counts.items()
        } if total_donations > 0 else {}
        
        return {
            "location_distribution": dict(location_counts),
            "location_preferences": location_preferences,
            "geographic_diversity": len(location_counts),
            "primary_locations": sorted(location_preferences.items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _analyze_campaign_size_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze preferences for campaign sizes."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        size_categories = {"small": 0, "medium": 0, "large": 0}
        
        for donation in user_donations:
            project = self.project_repo.get_by_id(donation.project_id)
            if project and hasattr(project, 'target_amount'):
                if project.target_amount < 10000000:  # < 10M IDR
                    size_categories["small"] += 1
                elif project.target_amount < 100000000:  # < 100M IDR
                    size_categories["medium"] += 1
                else:
                    size_categories["large"] += 1
        
        total = sum(size_categories.values())
        preferences = {
            size: count / total for size, count in size_categories.items()
        } if total > 0 else {}
        
        return {
            "size_distribution": size_categories,
            "size_preferences": preferences,
            "preferred_size": max(preferences.items(), key=lambda x: x[1])[0] if preferences else "unknown"
        }
    
    def _analyze_impact_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze user preferences for different impact types."""
        # This would require additional campaign metadata about impact types
        # For now, return a placeholder structure
        return {
            "impact_categories": {},
            "impact_preferences": {},
            "preferred_impact_type": "unknown"
        }
    
    # Social profile methods
    def _identify_user_cluster(self, user_id: str, similar_users: List[Dict]) -> Dict[str, Any]:
        """Identify which cluster the user belongs to."""
        if not similar_users:
            return {"cluster": "isolated", "cluster_size": 1}
        
        # Simple clustering based on similarity scores
        high_similarity_users = [u for u in similar_users if u.get('similarity', 0) > 0.7]
        
        if len(high_similarity_users) >= 3:
            cluster = "core_community"
        elif len(similar_users) >= 5:
            cluster = "extended_community"
        else:
            cluster = "small_group"
        
        return {
            "cluster": cluster,
            "cluster_size": len(similar_users),
            "high_similarity_count": len(high_similarity_users),
            "average_similarity": statistics.mean([u.get('similarity', 0) for u in similar_users])
        }
    
    def _analyze_user_influence(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's influence within the community."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        total_amount = sum(d.amount for d in user_donations)
        donation_count = len(user_donations)
        
        # Calculate influence metrics
        amount_influence = min(1.0, total_amount / 10000000)  # Normalize to 10M IDR
        frequency_influence = min(1.0, donation_count / 20)  # Normalize to 20 donations
        
        overall_influence = (amount_influence + frequency_influence) / 2
        
        if overall_influence >= 0.8:
            influence_level = "high"
        elif overall_influence >= 0.5:
            influence_level = "medium"
        else:
            influence_level = "low"
        
        return {
            "influence_level": influence_level,
            "influence_score": overall_influence,
            "amount_influence": amount_influence,
            "frequency_influence": frequency_influence,
            "total_contributed": total_amount,
            "community_position": self._determine_community_position(user_id)
        }
    
    def _determine_community_position(self, user_id: str) -> str:
        """Determine user's position in the community."""
        user_donations = self.donation_repo.get_by_user_id(user_id)
        
        if len(user_donations) >= 10:
            return "veteran"
        elif len(user_donations) >= 5:
            return "regular"
        elif len(user_donations) >= 2:
            return "emerging"
        else:
            return "newcomer"
    
    # Predictive methods
    def _predict_next_donation_amount(self, user_donations: List) -> Dict[str, Any]:
        """Predict the likely amount of the user's next donation."""
        if len(user_donations) < 2:
            return {"prediction": "insufficient_data", "confidence": 0}
        
        amounts = [d.amount for d in user_donations]
        
        # Calculate trend factor
        trend_factor = self._calculate_trend_factor(amounts)
        
        # Base prediction on recent average
        recent_amounts = amounts[-3:] if len(amounts) >= 3 else amounts
        base_amount = statistics.mean(recent_amounts)
        
        # Apply trend factor
        predicted_amount = base_amount * trend_factor
        
        # Calculate confidence based on consistency
        consistency = 1 - (statistics.stdev(amounts) / statistics.mean(amounts)) if statistics.mean(amounts) > 0 else 0
        confidence = max(0, min(1, consistency))
        
        return {
            "predicted_amount": predicted_amount,
            "confidence": confidence,
            "trend_factor": trend_factor,
            "base_amount": base_amount,
            "amount_range": {
                "min": predicted_amount * 0.7,
                "max": predicted_amount * 1.3
            }
        }
    
    def _calculate_trend_factor(self, amounts: List[float]) -> float:
        """Calculate trend factor for amount prediction."""
        if len(amounts) < 3:
            return 1.0
        
        # Simple trend calculation
        recent_avg = statistics.mean(amounts[-3:])
        historical_avg = statistics.mean(amounts[:-3]) if len(amounts) > 3 else statistics.mean(amounts)
        
        if historical_avg > 0:
            trend_factor = recent_avg / historical_avg
            # Limit extreme trends
            return max(0.5, min(2.0, trend_factor))
        
        return 1.0
    
    def _predict_category_interests(self, user_id: str) -> Dict[str, Any]:
        """Predict user's future category interests."""
        category_analysis = self._analyze_category_semantics(user_id)
        
        current_preferences = category_analysis.get("category_preferences", {})
        
        # Simple prediction based on current preferences
        # In practice, this could use more sophisticated ML models
        predicted_interests = {}
        for category, preference in current_preferences.items():
            # Predict slight increase in preferred categories
            predicted_interests[category] = min(1.0, preference * 1.1)
        
        return {
            "predicted_categories": predicted_interests,
            "confidence": 0.7,  # Placeholder confidence
            "recommendation": "Continue with preferred categories",
            "exploration_suggestions": list(current_preferences.keys())[:2]
        }
    
    def _calculate_donation_likelihood(self, user_donations: List) -> Dict[str, Any]:
        """Calculate likelihood of future donations."""
        if not user_donations:
            return {"likelihood": "low", "score": 0.1}
        
        # Calculate based on recent activity and patterns
        recent_activity = len([d for d in user_donations[-5:]]) if len(user_donations) >= 5 else len(user_donations)
        consistency_score = self._calculate_consistency_score(user_donations)
        
        likelihood_score = (recent_activity / 5 + consistency_score) / 2
        
        if likelihood_score >= 0.8:
            likelihood = "very_high"
        elif likelihood_score >= 0.6:
            likelihood = "high"
        elif likelihood_score >= 0.4:
            likelihood = "medium"
        else:
            likelihood = "low"
        
        return {
            "likelihood": likelihood,
            "score": likelihood_score,
            "recent_activity_factor": recent_activity / 5,
            "consistency_factor": consistency_score,
            "next_donation_timeframe": self._estimate_next_donation_timeframe(user_donations)
        }
    
    def _calculate_consistency_score(self, user_donations: List) -> float:
        """Calculate consistency score for donations."""
        if len(user_donations) < 2:
            return 0.5
        
        amounts = [d.amount for d in user_donations]
        cv = statistics.stdev(amounts) / statistics.mean(amounts) if statistics.mean(amounts) > 0 else 1
        
        return max(0, 1 - cv)
    
    def _estimate_next_donation_timeframe(self, user_donations: List) -> str:
        """Estimate when the next donation might occur."""
        if len(user_donations) < 2:
            return "unknown"
        
        # This would require actual timestamp analysis
        # For now, return based on donation frequency
        if len(user_donations) >= 10:
            return "within_month"
        elif len(user_donations) >= 5:
            return "within_quarter"
        else:
            return "within_year"
    
    def _assess_growth_potential(self, user_donations: List) -> Dict[str, Any]:
        """Assess user's potential for donation growth."""
        if not user_donations:
            return {"potential": "unknown", "score": 0.5}
        
        amounts = [d.amount for d in user_donations]
        trend = self._detect_amount_trend(amounts)
        
        # Calculate growth indicators
        if trend == "increasing":
            growth_score = 0.8
            potential = "high"
        elif trend == "stable" and len(user_donations) >= 5:
            growth_score = 0.6
            potential = "medium"
        else:
            growth_score = 0.4
            potential = "low"
        
        return {
            "potential": potential,
            "score": growth_score,
            "trend": trend,
            "current_trajectory": "positive" if trend == "increasing" else "stable",
            "growth_recommendations": self._generate_growth_recommendations(trend, amounts)
        }
    
    def _generate_growth_recommendations(self, trend: str, amounts: List[float]) -> List[str]:
        """Generate recommendations for donation growth."""
        recommendations = []
        
        if trend == "decreasing":
            recommendations.append("Re-engage with preferred categories")
            recommendations.append("Highlight impact of previous donations")
        elif trend == "stable":
            recommendations.append("Introduce new campaign categories")
            recommendations.append("Show community impact stories")
        else:  # increasing
            recommendations.append("Continue current engagement strategy")
            recommendations.append("Introduce premium campaign opportunities")
        
        return recommendations
    
    def _assess_churn_risk(self, user_donations: List) -> Dict[str, Any]:
        """Assess risk of user churning."""
        if not user_donations:
            return {"risk": "high", "score": 0.9}
        
        # Calculate churn indicators
        recent_activity = len(user_donations[-3:]) if len(user_donations) >= 3 else len(user_donations)
        total_activity = len(user_donations)
        
        # Simple churn risk calculation
        if recent_activity == 0:
            risk_score = 0.9
            risk_level = "very_high"
        elif recent_activity < total_activity * 0.3:
            risk_score = 0.7
            risk_level = "high"
        elif recent_activity < total_activity * 0.6:
            risk_score = 0.4
            risk_level = "medium"
        else:
            risk_score = 0.2
            risk_level = "low"
        
        return {
            "risk": risk_level,
            "score": risk_score,
            "recent_activity_ratio": recent_activity / max(1, total_activity),
            "retention_recommendations": self._generate_retention_recommendations(risk_level)
        }
    
    def _generate_retention_recommendations(self, risk_level: str) -> List[str]:
        """Generate recommendations for user retention."""
        recommendations = []
        
        if risk_level in ["high", "very_high"]:
            recommendations.extend([
                "Send personalized re-engagement campaign",
                "Highlight impact of previous donations",
                "Offer exclusive campaign access"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Increase communication frequency",
                "Share success stories from supported campaigns"
            ])
        else:
            recommendations.extend([
                "Maintain current engagement level",
                "Continue with personalized recommendations"
            ])
        
        return recommendations
    
    def _calculate_profile_completeness(self, base_profile: Dict[str, Any]) -> float:
        """Calculate how complete the user profile is."""
        required_fields = ['total_donations', 'total_amount', 'categories', 'avg_amount']
        present_fields = sum(1 for field in required_fields if field in base_profile and base_profile[field])
        
        return present_fields / len(required_fields)
    
    def _calculate_recommendation_confidence(self, base_profile: Dict[str, Any]) -> float:
        """Calculate confidence level for recommendations."""
        total_donations = base_profile.get('total_donations', 0)
        
        if total_donations >= 10:
            return 0.9
        elif total_donations >= 5:
            return 0.7
        elif total_donations >= 2:
            return 0.5
        else:
            return 0.3


def example_usage():
    """Example usage of the AdvancedUserProfilerDB."""
    from .database import DatabaseConfig, DatabaseManager
    
    # Initialize database
    config = DatabaseConfig()
    db_manager = DatabaseManager(config)
    
    # Create profiler
    profiler = AdvancedUserProfilerDB(db_manager)
    
    # Create comprehensive profile
    user_id = "1001"
    profile = profiler.create_comprehensive_profile(user_id)
    
    print(f"Comprehensive Profile for User {user_id}:")
    print(f"Behavioral Type: {profile.get('behavioral_profile', {}).get('behavioral_type', 'Unknown')}")
    print(f"Risk Tolerance: {profile.get('behavioral_profile', {}).get('risk_tolerance', {}).get('tolerance', 'Unknown')}")
    print(f"Engagement Level: {profile.get('behavioral_profile', {}).get('engagement_level', {}).get('level', 'Unknown')}")
    print(f"Profile Completeness: {profile.get('profile_metadata', {}).get('completeness_score', 0):.2f}")


if __name__ == "__main__":
    example_usage()