"""
Database-enabled donation analyzer for user profiling and recommendation.

This module provides comprehensive analysis of user donation patterns using database
instead of JSON files to improve recommendation quality for undonated projects.
"""

import numpy as np
import os
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import logging

from .models import Project, Donation, User, ProjectRepository, DonationRepository, UserRepository
from .database import init_database, get_db_manager

logger = logging.getLogger(__name__)


class DatabaseDonationAnalyzer:
    """Analyzes user donation patterns using database to extract insights for better recommendations."""
    
    def __init__(self, auto_init_db: bool = True):
        """Initialize the analyzer with database connection."""
        # Respect env gate to skip table creation
        if auto_init_db and os.getenv('DB_INIT_ENABLED', '1') != '0':
            try:
                init_database()
                logger.info("Database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
        
        # Initialize repositories using the shared DB manager
        dbm = get_db_manager()
        self.project_repo = ProjectRepository(dbm)
        self.donation_repo = DonationRepository(dbm)
        self.user_repo = UserRepository(dbm)
        
        # Cache for performance
        self._projects_cache = None
        self._project_lookup_cache = None
    
    def _get_projects(self) -> List[Project]:
        """Get all projects with caching."""
        if self._projects_cache is None:
            self._projects_cache = self.project_repo.get_all()
            logger.info(f"Loaded {len(self._projects_cache)} projects from database")
        return self._projects_cache
    
    def _get_project_lookup(self) -> Dict[str, Project]:
        """Get project lookup dictionary with caching."""
        if self._project_lookup_cache is None:
            projects = self._get_projects()
            self._project_lookup_cache = {c.id: c for c in projects}
        return self._project_lookup_cache
    
    def get_user_donation_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Create comprehensive donation profile for a user from database.
        
        Returns detailed analysis including:
        - Donation frequency and patterns
        - Category preferences with weights
        - Amount patterns and generosity level
        - Temporal patterns (when they donate)
        - Project type preferences
        """
        try:
            # Get user donations from database
            user_donations = self.donation_repo.get_by_user_id(user_id)
            # Contextual debug to reduce confusion with mixed logs
            try:
                sample_ids = [getattr(d, 'id', None) for d in (user_donations[:3] if isinstance(user_donations, list) else [])]
                logger.debug(
                    "Analyzer fetched donations: count=%d user_id=%s sample_ids=%s",
                    len(user_donations) if isinstance(user_donations, list) else 0,
                    user_id,
                    sample_ids,
                )
            except Exception:
                # Avoid breaking flow due to logging
                pass
            
            if not user_donations:
                logger.warning(f"DB No donations found for user {user_id}")
                return self._get_empty_profile()
            
            # Convert to dict format for compatibility with existing analysis methods
            donations_dict = [donation.to_dict() for donation in user_donations]
            
            # Enrich donations with project details
            enriched_donations = self._enrich_donations_with_projects(donations_dict)
            
            if not enriched_donations:
                logger.warning(f"No enriched donations found for user {user_id}")
                return self._get_empty_profile()
            
            # Perform comprehensive analysis
            profile = {
                'user_id': user_id,
                'total_donations': len(enriched_donations),
                'total_amount': sum(d['amount'] for d in enriched_donations),
                'basic_stats': self._calculate_basic_stats(enriched_donations),
                'category_analysis': self._analyze_categories(enriched_donations),
                'amount_analysis': self._analyze_amounts(enriched_donations),
                'temporal_analysis': self._analyze_temporal_patterns(enriched_donations),
                'project_characteristics': self._analyze_project_characteristics(enriched_donations),
                'generosity_score': self._calculate_generosity_score(enriched_donations),
                'diversity_score': self._calculate_diversity_score(enriched_donations),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Generated profile for user {user_id} with {len(enriched_donations)} donations")
            return profile
            
        except Exception as e:
            logger.error(f"Error generating profile for user {user_id}: {e}")
            return self._get_empty_profile()
    
    def _enrich_donations_with_projects(self, donations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich donations with project details from database."""
        enriched = []
        project_lookup = self._get_project_lookup()
        
        for donation in donations:
            project_id = str(donation.get('project_id') or donation.get('projects_id'))
            project = project_lookup.get(project_id)
            
            if project:
                enriched_donation = donation.copy()
                enriched_donation.update({
                    'project_title': project.title,
                    'project_category': getattr(project, 'category', 'Unknown'),
                    'project_target_amount': getattr(project, 'target_amount', 0),
                    'project_location': getattr(project, 'location', 'Unknown'),
                    'project_urgency_level': getattr(project, 'urgency_level', 'Medium'),
                    'project_type': getattr(project, 'project_type', 'General')
                })
                enriched.append(enriched_donation)
            else:
                logger.warning(f"Project {project_id} not found for donation {donation['id']}")
        
        return enriched
    
    def _get_empty_profile(self) -> Dict[str, Any]:
        """Return empty profile structure."""
        return {
            'user_id': '',
            'total_donations': 0,
            'total_amount': 0,
            'basic_stats': {},
            'category_analysis': {},
            'amount_analysis': {},
            'temporal_analysis': {},
            'project_characteristics': {},
            'generosity_score': 0.0,
            'diversity_score': 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_basic_stats(self, donations: List[Dict]) -> Dict[str, Any]:
        """Calculate basic donation statistics."""
        amounts = [d['amount'] for d in donations]
        
        return {
            'avg_donation': statistics.mean(amounts),
            'median_donation': statistics.median(amounts),
            'min_donation': min(amounts),
            'max_donation': max(amounts),
            'std_deviation': statistics.stdev(amounts) if len(amounts) > 1 else 0,
            'donation_frequency': len(donations)
        }
    
    def _analyze_categories(self, enriched_donations: List[Dict]) -> Dict[str, Any]:
        """Analyze category preferences with weights."""
        category_amounts = defaultdict(float)
        category_counts = defaultdict(int)
        
        total_amount = sum(d['amount'] for d in enriched_donations)
        
        for donation in enriched_donations:
            category = donation.get('project_category', 'Unknown')
            category_amounts[category] += donation['amount']
            category_counts[category] += 1
        
        # Calculate preferences based on both amount and frequency
        preferences = {}
        for category in category_amounts:
            amount_weight = category_amounts[category] / total_amount if total_amount > 0 else 0
            frequency_weight = category_counts[category] / len(enriched_donations)
            # Combined preference score (70% amount, 30% frequency)
            preferences[category] = (amount_weight * 0.7) + (frequency_weight * 0.3)
        
        return {
            'preferences': preferences,
            'category_amounts': dict(category_amounts),
            'category_counts': dict(category_counts),
            'top_category': max(preferences.keys(), key=preferences.get) if preferences else None
        }
    
    def _analyze_amounts(self, donations: List[Dict]) -> Dict[str, Any]:
        """Analyze donation amount patterns."""
        amounts = [d['amount'] for d in donations]
        
        return {
            'amount_distribution': self._categorize_amounts(amounts),
            'preferred_range': self._get_preferred_amount_range(amounts),
            'amount_trend': self._calculate_amount_trend(donations)
        }
    
    def _categorize_amounts(self, amounts: List[float]) -> Dict[str, int]:
        """Categorize amounts into ranges."""
        categories = {'small': 0, 'medium': 0, 'large': 0, 'very_large': 0}
        
        for amount in amounts:
            if amount < 50000:  # IDR 50k
                categories['small'] += 1
            elif amount < 200000:  # IDR 200k
                categories['medium'] += 1
            elif amount < 500000:  # IDR 500k
                categories['large'] += 1
            else:
                categories['very_large'] += 1
        
        return categories
    
    def _get_preferred_amount_range(self, amounts: List[float]) -> str:
        """Get user's preferred donation amount range."""
        if not amounts:
            return 'unknown'
        
        avg_amount = statistics.mean(amounts)
        
        if avg_amount < 50000:
            return 'small'
        elif avg_amount < 200000:
            return 'medium'
        elif avg_amount < 500000:
            return 'large'
        else:
            return 'very_large'
    
    def _calculate_amount_trend(self, donations: List[Dict]) -> str:
        """Calculate if donation amounts are increasing, decreasing, or stable."""
        if len(donations) < 3:
            return 'insufficient_data'
        
        # Sort by donation date
        sorted_donations = sorted(donations, key=lambda x: x.get('donation_date', ''))
        amounts = [d['amount'] for d in sorted_donations]
        
        # Calculate trend using linear regression slope
        x = list(range(len(amounts)))
        slope = np.polyfit(x, amounts, 1)[0]
        
        if slope > 10000:  # IDR 10k increase per donation
            return 'increasing'
        elif slope < -10000:  # IDR 10k decrease per donation
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_temporal_patterns(self, donations: List[Dict]) -> Dict[str, Any]:
        """Analyze when users typically donate."""
        # This is a simplified version - you can expand based on your needs
        return {
            'donation_frequency': len(donations),
            'recent_activity': len([d for d in donations 
                                  if self._is_recent_donation(d.get('donation_date'))]),
            'activity_level': self._calculate_activity_level(donations)
        }
    
    def _is_recent_donation(self, donation_date: str) -> bool:
        """Check if donation is recent (within last 6 months)."""
        if not donation_date:
            return False
        
        try:
            date = datetime.fromisoformat(donation_date.replace('Z', '+00:00'))
            return (datetime.now() - date).days <= 180
        except:
            return False
    
    def _calculate_activity_level(self, donations: List[Dict]) -> str:
        """Calculate user activity level."""
        if len(donations) >= 10:
            return 'high'
        elif len(donations) >= 5:
            return 'medium'
        elif len(donations) >= 1:
            return 'low'
        else:
            return 'inactive'
    
    def _analyze_project_characteristics(self, enriched_donations: List[Dict]) -> Dict[str, Any]:
        """Analyze preferred project characteristics."""
        urgency_preferences = Counter()
        type_preferences = Counter()
        location_preferences = Counter()
        
        for donation in enriched_donations:
            urgency_preferences[donation.get('project_urgency_level', 'medium')] += 1
            type_preferences[donation.get('project_type', 'general')] += 1
            location_preferences[donation.get('project_location', 'unknown')] += 1
        
        return {
            'urgency_preferences': dict(urgency_preferences),
            'type_preferences': dict(type_preferences),
            'location_preferences': dict(location_preferences),
            'preferred_urgency': urgency_preferences.most_common(1)[0][0] if urgency_preferences else 'medium',
            'preferred_type': type_preferences.most_common(1)[0][0] if type_preferences else 'general'
        }
    
    def _calculate_generosity_score(self, donations: List[Dict]) -> float:
        """Calculate user generosity score (0-1)."""
        if not donations:
            return 0.0
        
        total_amount = sum(d['amount'] for d in donations)
        donation_count = len(donations)
        
        # Normalize based on amount and frequency
        # This is a simplified calculation - adjust based on your business logic
        amount_score = min(total_amount / 1000000, 1.0)  # Cap at IDR 1M
        frequency_score = min(donation_count / 20, 1.0)  # Cap at 20 donations
        
        return (amount_score * 0.6) + (frequency_score * 0.4)
    
    def _calculate_diversity_score(self, enriched_donations: List[Dict]) -> float:
        """Calculate donation diversity score (0-1)."""
        if not enriched_donations:
            return 0.0
        
        categories = set(d.get('project_category', 'Unknown') for d in enriched_donations)
        types = set(d.get('project_type', 'general') for d in enriched_donations)
        
        # Calculate diversity based on unique categories and types
        category_diversity = len(categories) / 10  # Assuming max 10 categories
        type_diversity = len(types) / 5  # Assuming max 5 types
        
        return min((category_diversity + type_diversity) / 2, 1.0)
    
    def get_undonated_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get projects that user hasn't donated to as dictionaries.
        
        Downstream components (smart filters and comprehensive recommender) expect
        project items to be dictionary-like. Convert Project dataclass instances
        to dicts to ensure compatibility.
        """
        try:
            # Get user's donated project IDs
            user_donations = self.donation_repo.get_by_user_id(user_id)
            donated_project_ids = set(donation.project_id for donation in user_donations)
            
            # Get all projects and filter out donated ones
            all_projects = self._get_projects()
            undonated_projects = [
                project.to_dict() for project in all_projects 
                if project.id not in donated_project_ids
            ]
            
            logger.info(f"Found {len(undonated_projects)} undonated projects for user {user_id}")
            return undonated_projects
            
        except Exception as e:
            logger.error(f"Error getting undonated projects for user {user_id}: {e}")
            return []
    
    def find_similar_users(self, target_user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find users with similar donation patterns."""
        try:
            print('calling find_similar_users')
            target_profile = self.get_user_donation_profile(target_user_id)
            
            if target_profile['total_donations'] == 0:
                logger.warning(f"No donations found for target user {target_user_id}")
                return []
            
            # Get all users who have made donations
            all_donations = self.donation_repo.get_all()
            user_ids = set(donation.user_id for donation in all_donations)
            user_ids.discard(target_user_id)  # Remove target user
            
            similarities = []
            
            for user_id in user_ids:
                user_profile = self.get_user_donation_profile(user_id)
                if user_profile['total_donations'] > 0:
                    similarity = self._calculate_user_similarity(target_profile, user_profile)
                    similarities.append({
                        'user_id': user_id,
                        'similarity_score': similarity,
                        'profile': user_profile
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(similarities[:top_k])} similar users for {target_user_id}")
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar users for {target_user_id}: {e}")
            return []
    
    def _calculate_user_similarity(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
        """Calculate similarity between two user profiles."""
        try:
            # Category preference similarity
            cat_sim = self._calculate_category_similarity(
                profile1.get('category_analysis', {}).get('preferences', {}),
                profile2.get('category_analysis', {}).get('preferences', {})
            )
            
            # Amount range similarity
            amount_sim = self._calculate_amount_similarity(
                profile1.get('amount_analysis', {}),
                profile2.get('amount_analysis', {})
            )
            
            # Generosity similarity
            gen_sim = 1 - abs(profile1.get('generosity_score', 0) - profile2.get('generosity_score', 0))
            
            # Combined similarity (weighted average)
            return (cat_sim * 0.5) + (amount_sim * 0.3) + (gen_sim * 0.2)
            
        except Exception as e:
            logger.error(f"Error calculating user similarity: {e}")
            return 0.0
    
    def _calculate_category_similarity(self, prefs1: Dict[str, float], prefs2: Dict[str, float]) -> float:
        """Calculate similarity between category preferences."""
        if not prefs1 or not prefs2:
            return 0.0
        
        all_categories = set(prefs1.keys()) | set(prefs2.keys())
        
        if not all_categories:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(prefs1.get(cat, 0) * prefs2.get(cat, 0) for cat in all_categories)
        norm1 = sum(val ** 2 for val in prefs1.values()) ** 0.5
        norm2 = sum(val ** 2 for val in prefs2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_amount_similarity(self, amount1: Dict[str, Any], amount2: Dict[str, Any]) -> float:
        """Calculate similarity between amount patterns."""
        range1 = amount1.get('preferred_range', 'unknown')
        range2 = amount2.get('preferred_range', 'unknown')
        
        if range1 == range2:
            return 1.0
        elif range1 == 'unknown' or range2 == 'unknown':
            return 0.0
        else:
            # Adjacent ranges get partial similarity
            ranges = ['small', 'medium', 'large', 'very_large']
            try:
                idx1, idx2 = ranges.index(range1), ranges.index(range2)
                distance = abs(idx1 - idx2)
                return max(0, 1 - (distance / len(ranges)))
            except ValueError:
                return 0.0


def example_usage():
    """Example usage of the DatabaseDonationAnalyzer."""
    try:
        # Initialize analyzer
        analyzer = DatabaseDonationAnalyzer()
        
        # Analyze a user
        user_id = "1001"
        print('calling create example usage')
        profile = analyzer.get_user_donation_profile(user_id)
        
        print(f"User Profile for {user_id}:")
        print(f"Total Donations: {profile['total_donations']}")
        print(f"Total Amount: IDR {profile['total_amount']:,.0f}")
        print(f"Generosity Score: {profile['generosity_score']:.2f}")
        print(f"Diversity Score: {profile['diversity_score']:.2f}")
        
        if profile['category_analysis'].get('preferences'):
            print("\nTop Categories:")
            for cat, score in sorted(profile['category_analysis']['preferences'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {cat}: {score:.3f}")
        
        # Find similar users
        similar_users = analyzer.find_similar_users(user_id, top_k=3)
        print(f"\nSimilar Users:")
        for user in similar_users:
            print(f"  User {user['user_id']}: {user['similarity_score']:.3f}")
        
        # Get undonated projects
        undonated = analyzer.get_undonated_projects(user_id)
        print(f"\nUndonated Projects: {len(undonated)}")
        
    except Exception as e:
        print(f"Error in example usage: {e}")


if __name__ == "__main__":
    example_usage()