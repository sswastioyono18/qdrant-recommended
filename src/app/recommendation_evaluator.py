"""
Recommendation Evaluation and Testing Framework

This module provides comprehensive evaluation metrics and testing capabilities
for the recommendation system, including:
1. Accuracy and relevance metrics
2. Diversity and coverage analysis
3. Performance benchmarking
4. A/B testing framework
5. User satisfaction simulation
6. Business impact metrics
"""

import json
import numpy as np
import statistics
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
import random
import time
import sys
import os

# Add the parent directory to the path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_recommender import ComprehensiveRecommendationEngine
from smart_filter import SmartCampaignFilter
from advanced_profiler import AdvancedUserProfiler
from donation_analyzer import DonationAnalyzer


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.
    """
    
    def __init__(self, donations_path: str = "../../data/donations.json", 
                 campaigns_path: str = "../../data/campaigns.json"):
        """Initialize the evaluation framework."""
        self.donations_path = donations_path
        self.campaigns_path = campaigns_path
        
        # Initialize systems to evaluate
        self.comprehensive_engine = ComprehensiveRecommendationEngine(donations_path, campaigns_path)
        self.smart_filter = SmartCampaignFilter(donations_path, campaigns_path)
        self.profiler = AdvancedUserProfiler(donations_path, campaigns_path)
        self.analyzer = DonationAnalyzer(donations_path, campaigns_path)
        
        # Load data
        self.donations = self.analyzer.donations
        self.campaigns = self.analyzer.campaigns
        self.campaign_lookup = self.analyzer.campaign_lookup
        
        # Evaluation metrics storage
        self.evaluation_results = {}
    
    def run_comprehensive_evaluation(self, test_users: Optional[List[str]] = None,
                                   max_recommendations: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across multiple metrics.
        
        Args:
            test_users: List of user IDs to test (if None, uses all users)
            max_recommendations: Maximum recommendations per user
        
        Returns:
            Comprehensive evaluation results
        """
        print("🧪 Starting Comprehensive Recommendation System Evaluation...")
        print("=" * 80)
        
        # Prepare test users
        if test_users is None:
            test_users = list(self.donations.keys())[:10]  # Limit for demo
        
        print(f"📊 Evaluating {len(test_users)} users with up to {max_recommendations} recommendations each")
        
        # Run evaluations
        start_time = time.time()
        
        results = {
            "evaluation_metadata": {
                "test_users_count": len(test_users),
                "max_recommendations": max_recommendations,
                "evaluation_timestamp": time.time(),
                "test_users": test_users
            },
            "accuracy_metrics": self._evaluate_accuracy(test_users, max_recommendations),
            "diversity_metrics": self._evaluate_diversity(test_users, max_recommendations),
            "coverage_metrics": self._evaluate_coverage(test_users, max_recommendations),
            "performance_metrics": self._evaluate_performance(test_users, max_recommendations),
            "user_satisfaction_simulation": self._simulate_user_satisfaction(test_users, max_recommendations),
            "business_impact_metrics": self._evaluate_business_impact(test_users, max_recommendations),
            "system_comparison": self._compare_recommendation_approaches(test_users[:5])  # Smaller sample for comparison
        }
        
        evaluation_time = time.time() - start_time
        results["evaluation_metadata"]["evaluation_duration_seconds"] = evaluation_time
        
        print(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Store results
        self.evaluation_results = results
        
        return results
    
    def _evaluate_accuracy(self, test_users: List[str], max_recommendations: int) -> Dict[str, Any]:
        """Evaluate recommendation accuracy and relevance."""
        print("🎯 Evaluating Accuracy Metrics...")
        
        accuracy_scores = []
        relevance_scores = []
        precision_scores = []
        
        for user_id in test_users:
            try:
                # Get recommendations
                recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations, include_explanations=False
                )
                
                if "error" in recommendations:
                    continue
                
                # Calculate accuracy based on user profile alignment
                user_profile = self.profiler.create_comprehensive_profile(user_id)
                if "error" in user_profile:
                    continue
                
                # Accuracy: How well recommendations match user preferences
                user_categories = set(user_profile.get('category_preferences', {}).get('preferences', {}).keys())
                rec_categories = set(rec.get('category_name', 'Unknown') for rec in recommendations['recommendations'])
                
                if user_categories and rec_categories:
                    category_overlap = len(user_categories.intersection(rec_categories)) / len(user_categories)
                else:
                    category_overlap = 0.5  # Neutral for unknown categories
                
                accuracy_scores.append(category_overlap)
                
                # Relevance: Average recommendation scores
                if recommendations['recommendations']:
                    avg_relevance = statistics.mean(rec['final_score'] for rec in recommendations['recommendations'])
                    relevance_scores.append(avg_relevance)
                
                # Precision: Percentage of high-quality recommendations (score > 0.6)
                high_quality_recs = sum(1 for rec in recommendations['recommendations'] if rec['final_score'] > 0.6)
                precision = high_quality_recs / len(recommendations['recommendations']) if recommendations['recommendations'] else 0
                precision_scores.append(precision)
                
            except Exception as e:
                print(f"⚠️ Error evaluating user {user_id}: {e}")
                continue
        
        return {
            "category_alignment_accuracy": {
                "mean": statistics.mean(accuracy_scores) if accuracy_scores else 0,
                "median": statistics.median(accuracy_scores) if accuracy_scores else 0,
                "std_dev": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0,
                "sample_size": len(accuracy_scores)
            },
            "recommendation_relevance": {
                "mean_score": statistics.mean(relevance_scores) if relevance_scores else 0,
                "median_score": statistics.median(relevance_scores) if relevance_scores else 0,
                "std_dev": statistics.stdev(relevance_scores) if len(relevance_scores) > 1 else 0,
                "sample_size": len(relevance_scores)
            },
            "precision_metrics": {
                "mean_precision": statistics.mean(precision_scores) if precision_scores else 0,
                "high_precision_users": sum(1 for p in precision_scores if p > 0.7),
                "low_precision_users": sum(1 for p in precision_scores if p < 0.3),
                "sample_size": len(precision_scores)
            }
        }
    
    def _evaluate_diversity(self, test_users: List[str], max_recommendations: int) -> Dict[str, Any]:
        """Evaluate recommendation diversity and novelty."""
        print("🎨 Evaluating Diversity Metrics...")
        
        all_recommendations = []
        user_diversity_scores = []
        category_distributions = []
        
        for user_id in test_users:
            try:
                recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations, include_explanations=False
                )
                
                if "error" in recommendations or not recommendations['recommendations']:
                    continue
                
                user_recs = recommendations['recommendations']
                all_recommendations.extend(user_recs)
                
                # Calculate intra-user diversity (category diversity within user's recommendations)
                user_categories = [rec.get('category_name', 'Unknown') for rec in user_recs]
                unique_categories = len(set(user_categories))
                total_categories = len(user_categories)
                
                diversity_score = unique_categories / total_categories if total_categories > 0 else 0
                user_diversity_scores.append(diversity_score)
                
                # Track category distribution
                category_counter = Counter(user_categories)
                category_distributions.append(category_counter)
                
            except Exception as e:
                print(f"⚠️ Error evaluating diversity for user {user_id}: {e}")
                continue
        
        # Calculate inter-user diversity (how different recommendations are across users)
        all_campaign_ids = [rec['campaign_id'] for rec in all_recommendations]
        unique_campaigns = len(set(all_campaign_ids))
        total_recommendations = len(all_campaign_ids)
        
        # Calculate category distribution across all recommendations
        all_categories = [rec.get('category_name', 'Unknown') for rec in all_recommendations]
        category_distribution = Counter(all_categories)
        
        # Calculate Gini coefficient for category distribution (inequality measure)
        gini_coefficient = self._calculate_gini_coefficient(list(category_distribution.values()))
        
        return {
            "intra_user_diversity": {
                "mean_diversity": statistics.mean(user_diversity_scores) if user_diversity_scores else 0,
                "median_diversity": statistics.median(user_diversity_scores) if user_diversity_scores else 0,
                "high_diversity_users": sum(1 for d in user_diversity_scores if d > 0.7),
                "low_diversity_users": sum(1 for d in user_diversity_scores if d < 0.3),
                "sample_size": len(user_diversity_scores)
            },
            "inter_user_diversity": {
                "unique_campaigns_recommended": unique_campaigns,
                "total_recommendations": total_recommendations,
                "campaign_diversity_ratio": unique_campaigns / total_recommendations if total_recommendations > 0 else 0
            },
            "category_distribution": {
                "categories_represented": len(category_distribution),
                "category_counts": dict(category_distribution),
                "gini_coefficient": gini_coefficient,
                "distribution_balance": "balanced" if gini_coefficient < 0.3 else "moderate" if gini_coefficient < 0.6 else "concentrated"
            }
        }
    
    def _evaluate_coverage(self, test_users: List[str], max_recommendations: int) -> Dict[str, Any]:
        """Evaluate system coverage and reach."""
        print("📈 Evaluating Coverage Metrics...")
        
        recommended_campaigns = set()
        total_campaigns = len(self.campaigns)
        user_coverage_scores = []
        
        for user_id in test_users:
            try:
                recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations, include_explanations=False
                )
                
                if "error" in recommendations:
                    continue
                
                user_campaign_ids = set(rec['campaign_id'] for rec in recommendations['recommendations'])
                recommended_campaigns.update(user_campaign_ids)
                
                # Calculate user-specific coverage (how many available campaigns were considered)
                user_donated_campaigns = set(str(d['campaign_id']) for d in self.donations.get(user_id, []))
                available_campaigns = total_campaigns - len(user_donated_campaigns)
                
                if available_campaigns > 0:
                    user_coverage = len(user_campaign_ids) / min(available_campaigns, max_recommendations)
                    user_coverage_scores.append(user_coverage)
                
            except Exception as e:
                print(f"⚠️ Error evaluating coverage for user {user_id}: {e}")
                continue
        
        # Calculate category coverage
        recommended_categories = set()
        all_categories = set()
        
        for campaign in self.campaigns:
            category = campaign.get('category_name', 'Unknown')
            all_categories.add(category)
            
            if campaign['id'] in recommended_campaigns:
                recommended_categories.add(category)
        
        return {
            "campaign_coverage": {
                "campaigns_recommended": len(recommended_campaigns),
                "total_campaigns": total_campaigns,
                "coverage_percentage": len(recommended_campaigns) / total_campaigns * 100 if total_campaigns > 0 else 0
            },
            "category_coverage": {
                "categories_recommended": len(recommended_categories),
                "total_categories": len(all_categories),
                "coverage_percentage": len(recommended_categories) / len(all_categories) * 100 if all_categories else 0
            },
            "user_coverage_efficiency": {
                "mean_coverage": statistics.mean(user_coverage_scores) if user_coverage_scores else 0,
                "median_coverage": statistics.median(user_coverage_scores) if user_coverage_scores else 0,
                "sample_size": len(user_coverage_scores)
            }
        }
    
    def _evaluate_performance(self, test_users: List[str], max_recommendations: int) -> Dict[str, Any]:
        """Evaluate system performance metrics."""
        print("⚡ Evaluating Performance Metrics...")
        
        response_times = []
        memory_usage = []
        success_rates = []
        
        for user_id in test_users:
            try:
                start_time = time.time()
                
                recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations, include_explanations=True
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # Track success/failure
                if "error" not in recommendations:
                    success_rates.append(1)
                else:
                    success_rates.append(0)
                
            except Exception as e:
                print(f"⚠️ Performance error for user {user_id}: {e}")
                success_rates.append(0)
                continue
        
        return {
            "response_time": {
                "mean_seconds": statistics.mean(response_times) if response_times else 0,
                "median_seconds": statistics.median(response_times) if response_times else 0,
                "max_seconds": max(response_times) if response_times else 0,
                "min_seconds": min(response_times) if response_times else 0,
                "sample_size": len(response_times)
            },
            "success_rate": {
                "success_percentage": statistics.mean(success_rates) * 100 if success_rates else 0,
                "successful_requests": sum(success_rates),
                "total_requests": len(success_rates),
                "failure_rate": (1 - statistics.mean(success_rates)) * 100 if success_rates else 100
            },
            "throughput": {
                "requests_per_second": len(test_users) / sum(response_times) if response_times and sum(response_times) > 0 else 0,
                "total_evaluation_time": sum(response_times) if response_times else 0
            }
        }
    
    def _simulate_user_satisfaction(self, test_users: List[str], max_recommendations: int) -> Dict[str, Any]:
        """Simulate user satisfaction based on recommendation quality."""
        print("😊 Simulating User Satisfaction...")
        
        satisfaction_scores = []
        engagement_predictions = []
        
        for user_id in test_users:
            try:
                recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations, include_explanations=False
                )
                
                if "error" in recommendations:
                    continue
                
                # Simulate satisfaction based on multiple factors
                user_profile = self.profiler.create_comprehensive_profile(user_id)
                if "error" in user_profile:
                    continue
                
                # Factor 1: Recommendation quality (average score)
                if recommendations['recommendations']:
                    avg_score = statistics.mean(rec['final_score'] for rec in recommendations['recommendations'])
                else:
                    avg_score = 0
                
                # Factor 2: Diversity satisfaction
                categories = [rec.get('category_name', 'Unknown') for rec in recommendations['recommendations']]
                diversity_satisfaction = len(set(categories)) / len(categories) if categories else 0
                
                # Factor 3: Profile confidence
                profile_confidence = user_profile.get('recommendation_confidence', 0.5)
                
                # Factor 4: Personalization level
                behavioral_type = user_profile.get('behavioral_profile', {}).get('behavioral_type', 'unknown')
                personalization_bonus = 0.1 if behavioral_type in ['loyal_consistent', 'loyal_variable'] else 0
                
                # Combine factors for overall satisfaction
                satisfaction = (
                    avg_score * 0.4 +
                    diversity_satisfaction * 0.2 +
                    profile_confidence * 0.3 +
                    personalization_bonus * 0.1
                )
                
                satisfaction_scores.append(satisfaction)
                
                # Predict engagement likelihood
                engagement_likelihood = satisfaction * 0.8 + random.uniform(0, 0.2)  # Add some randomness
                engagement_predictions.append(engagement_likelihood)
                
            except Exception as e:
                print(f"⚠️ Error simulating satisfaction for user {user_id}: {e}")
                continue
        
        return {
            "satisfaction_metrics": {
                "mean_satisfaction": statistics.mean(satisfaction_scores) if satisfaction_scores else 0,
                "median_satisfaction": statistics.median(satisfaction_scores) if satisfaction_scores else 0,
                "high_satisfaction_users": sum(1 for s in satisfaction_scores if s > 0.7),
                "low_satisfaction_users": sum(1 for s in satisfaction_scores if s < 0.4),
                "sample_size": len(satisfaction_scores)
            },
            "engagement_predictions": {
                "mean_engagement_likelihood": statistics.mean(engagement_predictions) if engagement_predictions else 0,
                "high_engagement_predicted": sum(1 for e in engagement_predictions if e > 0.7),
                "low_engagement_predicted": sum(1 for e in engagement_predictions if e < 0.4),
                "sample_size": len(engagement_predictions)
            }
        }
    
    def _evaluate_business_impact(self, test_users: List[str], max_recommendations: int) -> Dict[str, Any]:
        """Evaluate potential business impact metrics."""
        print("💰 Evaluating Business Impact...")
        
        potential_donations = []
        campaign_exposure_values = []
        
        for user_id in test_users:
            try:
                recommendations = self.comprehensive_engine.get_comprehensive_recommendations(
                    user_id, max_recommendations, include_explanations=False
                )
                
                if "error" in recommendations:
                    continue
                
                # Calculate potential donation value based on user history
                user_donations = self.donations.get(user_id, [])
                if user_donations:
                    avg_donation = statistics.mean(d.get('amount', 0) for d in user_donations)
                    total_donated = sum(d.get('amount', 0) for d in user_donations)
                else:
                    avg_donation = 50000  # Default assumption for new users
                    total_donated = 0
                
                # Estimate potential value from recommendations
                for rec in recommendations['recommendations']:
                    # Probability of donation based on recommendation score
                    donation_probability = rec['final_score']
                    
                    # Expected donation amount (based on user history and campaign target)
                    campaign_target = rec.get('target_amount', 0)
                    if campaign_target > 0:
                        expected_donation = min(avg_donation, campaign_target * 0.1)  # Assume 10% of target max
                    else:
                        expected_donation = avg_donation
                    
                    potential_value = donation_probability * expected_donation
                    potential_donations.append(potential_value)
                    
                    # Campaign exposure value (visibility benefit)
                    exposure_value = rec['final_score'] * 10000  # Arbitrary exposure value
                    campaign_exposure_values.append(exposure_value)
                
            except Exception as e:
                print(f"⚠️ Error evaluating business impact for user {user_id}: {e}")
                continue
        
        return {
            "potential_donation_value": {
                "total_potential_idr": sum(potential_donations),
                "mean_potential_per_recommendation": statistics.mean(potential_donations) if potential_donations else 0,
                "median_potential_per_recommendation": statistics.median(potential_donations) if potential_donations else 0,
                "sample_size": len(potential_donations)
            },
            "campaign_exposure_value": {
                "total_exposure_value": sum(campaign_exposure_values),
                "mean_exposure_per_recommendation": statistics.mean(campaign_exposure_values) if campaign_exposure_values else 0,
                "sample_size": len(campaign_exposure_values)
            },
            "roi_estimates": {
                "potential_revenue_idr": sum(potential_donations),
                "estimated_conversion_rate": 0.15,  # Assume 15% conversion rate
                "expected_actual_donations_idr": sum(potential_donations) * 0.15
            }
        }
    
    def _compare_recommendation_approaches(self, test_users: List[str]) -> Dict[str, Any]:
        """Compare different recommendation approaches."""
        print("🔄 Comparing Recommendation Approaches...")
        
        approaches = {
            "comprehensive": self.comprehensive_engine,
            "smart_filter_only": self.smart_filter
        }
        
        comparison_results = {}
        
        for approach_name, system in approaches.items():
            print(f"  Testing {approach_name}...")
            
            scores = []
            response_times = []
            
            for user_id in test_users:
                try:
                    start_time = time.time()
                    
                    if approach_name == "comprehensive":
                        result = system.get_comprehensive_recommendations(user_id, 5, False)
                        if "error" not in result and result['recommendations']:
                            avg_score = statistics.mean(rec['final_score'] for rec in result['recommendations'])
                            scores.append(avg_score)
                    else:
                        # For smart filter only, we need to simulate a simpler approach
                        user_profile = self.profiler.create_comprehensive_profile(user_id)
                        if "error" not in user_profile:
                            # Get undonated campaigns
                            donated_campaigns = set(str(d['campaign_id']) for d in self.donations.get(user_id, []))
                            undonated = [c for c in self.campaigns if str(c['id']) not in donated_campaigns][:5]
                            
                            if undonated:
                                # Calculate simple relevance scores
                                relevance_scores = []
                                for campaign in undonated:
                                    relevance, _ = system._calculate_campaign_relevance(user_profile, campaign, False)
                                    relevance_scores.append(relevance)
                                
                                if relevance_scores:
                                    avg_score = statistics.mean(relevance_scores)
                                    scores.append(avg_score)
                    
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                except Exception as e:
                    print(f"    ⚠️ Error with {approach_name} for user {user_id}: {e}")
                    continue
            
            comparison_results[approach_name] = {
                "mean_score": statistics.mean(scores) if scores else 0,
                "mean_response_time": statistics.mean(response_times) if response_times else 0,
                "sample_size": len(scores)
            }
        
        return comparison_results
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for measuring inequality."""
        if not values or len(values) == 0:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(sorted_values))
    
    def generate_evaluation_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate a comprehensive evaluation report."""
        if results is None:
            results = self.evaluation_results
        
        if not results:
            return "No evaluation results available. Run evaluation first."
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 80)
        
        # Metadata
        metadata = results.get('evaluation_metadata', {})
        report.append(f"\n📊 EVALUATION OVERVIEW")
        report.append(f"  • Test Users: {metadata.get('test_users_count', 0)}")
        report.append(f"  • Max Recommendations per User: {metadata.get('max_recommendations', 0)}")
        report.append(f"  • Evaluation Duration: {metadata.get('evaluation_duration_seconds', 0):.2f} seconds")
        
        # Accuracy Metrics
        accuracy = results.get('accuracy_metrics', {})
        report.append(f"\n🎯 ACCURACY METRICS")
        
        category_alignment = accuracy.get('category_alignment_accuracy', {})
        report.append(f"  • Category Alignment: {category_alignment.get('mean', 0):.3f} ± {category_alignment.get('std_dev', 0):.3f}")
        
        relevance = accuracy.get('recommendation_relevance', {})
        report.append(f"  • Average Relevance Score: {relevance.get('mean_score', 0):.3f}")
        
        precision = accuracy.get('precision_metrics', {})
        report.append(f"  • Mean Precision: {precision.get('mean_precision', 0):.3f}")
        report.append(f"  • High Precision Users: {precision.get('high_precision_users', 0)}")
        
        # Diversity Metrics
        diversity = results.get('diversity_metrics', {})
        report.append(f"\n🎨 DIVERSITY METRICS")
        
        intra_diversity = diversity.get('intra_user_diversity', {})
        report.append(f"  • Mean Intra-User Diversity: {intra_diversity.get('mean_diversity', 0):.3f}")
        
        inter_diversity = diversity.get('inter_user_diversity', {})
        report.append(f"  • Campaign Diversity Ratio: {inter_diversity.get('campaign_diversity_ratio', 0):.3f}")
        
        category_dist = diversity.get('category_distribution', {})
        report.append(f"  • Category Distribution Balance: {category_dist.get('distribution_balance', 'unknown')}")
        
        # Coverage Metrics
        coverage = results.get('coverage_metrics', {})
        report.append(f"\n📈 COVERAGE METRICS")
        
        campaign_coverage = coverage.get('campaign_coverage', {})
        report.append(f"  • Campaign Coverage: {campaign_coverage.get('coverage_percentage', 0):.1f}%")
        
        category_coverage = coverage.get('category_coverage', {})
        report.append(f"  • Category Coverage: {category_coverage.get('coverage_percentage', 0):.1f}%")
        
        # Performance Metrics
        performance = results.get('performance_metrics', {})
        report.append(f"\n⚡ PERFORMANCE METRICS")
        
        response_time = performance.get('response_time', {})
        report.append(f"  • Mean Response Time: {response_time.get('mean_seconds', 0):.3f} seconds")
        
        success_rate = performance.get('success_rate', {})
        report.append(f"  • Success Rate: {success_rate.get('success_percentage', 0):.1f}%")
        
        throughput = performance.get('throughput', {})
        report.append(f"  • Throughput: {throughput.get('requests_per_second', 0):.2f} requests/second")
        
        # User Satisfaction
        satisfaction = results.get('user_satisfaction_simulation', {})
        report.append(f"\n😊 USER SATISFACTION SIMULATION")
        
        satisfaction_metrics = satisfaction.get('satisfaction_metrics', {})
        report.append(f"  • Mean Satisfaction: {satisfaction_metrics.get('mean_satisfaction', 0):.3f}")
        report.append(f"  • High Satisfaction Users: {satisfaction_metrics.get('high_satisfaction_users', 0)}")
        
        engagement = satisfaction.get('engagement_predictions', {})
        report.append(f"  • Mean Engagement Likelihood: {engagement.get('mean_engagement_likelihood', 0):.3f}")
        
        # Business Impact
        business = results.get('business_impact_metrics', {})
        report.append(f"\n💰 BUSINESS IMPACT METRICS")
        
        potential_value = business.get('potential_donation_value', {})
        report.append(f"  • Total Potential Donations: IDR {potential_value.get('total_potential_idr', 0):,.0f}")
        
        roi = business.get('roi_estimates', {})
        report.append(f"  • Expected Actual Donations: IDR {roi.get('expected_actual_donations_idr', 0):,.0f}")
        
        # System Comparison
        comparison = results.get('system_comparison', {})
        if comparison:
            report.append(f"\n🔄 SYSTEM COMPARISON")
            for approach, metrics in comparison.items():
                report.append(f"  • {approach.replace('_', ' ').title()}:")
                report.append(f"    - Mean Score: {metrics.get('mean_score', 0):.3f}")
                report.append(f"    - Response Time: {metrics.get('mean_response_time', 0):.3f}s")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def example_usage():
    """Demonstrate the evaluation framework."""
    print("🧪 RECOMMENDATION SYSTEM EVALUATION FRAMEWORK DEMO")
    print("=" * 80)
    
    evaluator = RecommendationEvaluator()
    
    # Run comprehensive evaluation on a subset of users
    test_users = ["1001", "2002", "3003", "4004", "5005"]
    
    print(f"Running evaluation on users: {test_users}")
    
    results = evaluator.run_comprehensive_evaluation(
        test_users=test_users,
        max_recommendations=5
    )
    
    # Generate and display report
    print("\n" + evaluator.generate_evaluation_report(results))
    
    # Save results to file
    output_file = "../../data/evaluation_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Evaluation results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")


if __name__ == "__main__":
    example_usage()