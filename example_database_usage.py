#!/usr/bin/env python3
"""
Example script demonstrating how to use the recommendation system with database integration.

This script shows:
1. How to set up database connection
2. How to migrate from JSON to database
3. How to use the database-enabled recommendation system
4. How to get recommendations and user insights
"""

import os
import sys
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.app.database import DatabaseManager, DatabaseConfig
from src.app.comprehensive_recommender_db import ComprehensiveRecommendationEngineDB
from src.app.integration_guide_db import EnhancedRecommendationSystemDB


def setup_database_example():
    """Example of setting up database connection"""
    print("🗄️ Setting up database connection...")
    
    # Example 1: SQLite (for development)
    sqlite_config = {
        'type': 'sqlite',
        'database': 'recommendations.db'
    }
    
    # Example 2: PostgreSQL (for production)
    postgres_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'recommendations',
        'username': 'user',
        'password': 'password'
    }
    
    # Example 3: MySQL
    mysql_config = {
        'type': 'mysql',
        'host': 'proxysql.stg.kt.bs',
        'port': 6033,
        'database': 'kitabisa',
        'username': 'dewa',
        'password': 'm66FsuRgZB4AfKKUP2Ed'
    }
    
    # Use SQLite for this example
    db_manager = DatabaseManager(mysql_config)
    print("✅ Database connection established!")
    return db_manager


def migrate_json_data_example():
    """Example of migrating from JSON files to database"""
    print("\n📦 Migrating JSON data to database...")
    
    # Check if JSON files exist
    campaigns_file = "data/campaigns.json"
    donations_file = "data/donations.json"
    
    if not os.path.exists(campaigns_file) or not os.path.exists(donations_file):
        print("⚠️ JSON files not found. Skipping migration.")
        return
    
    # Database configuration
    db_config = {
        'type': 'sqlite',
        'database': 'recommendations.db'
    }
    
    try:
        # Perform migration
        migrate_json_to_database(
            campaigns_file=campaigns_file,
            donations_file=donations_file,
            db_config=db_config
        )
        print("✅ Migration completed successfully!")
    except Exception as e:
        print(f"❌ Migration failed: {e}")


def recommendation_system_example(db_manager: DatabaseManager):
    """Example of using the database-enabled recommendation system"""
    print("\n🎯 Using database-enabled recommendation system...")
    
    try:
        # Initialize the recommendation system
        recommender = EnhancedRecommendationSystemDB(db_manager)
        
        # Example user ID (you can change this)
        user_id = 569153655
        
        print(f"\n👤 Getting recommendations for user {user_id}...")
        
        # Get enhanced recommendations
        recommendations = recommender.get_enhanced_recommendations(
            user_id=user_id,
            max_recommendations=5,
            include_explanations=True
        )
        
        print(f"📋 Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. Campaign: {rec.get('title', 'N/A')}")
            print(f"   Score: {rec.get('final_score', 0):.3f}")
            print(f"   Category: {rec.get('category', 'N/A')}")
            print(f"   Target: ${rec.get('target_amount', 0):,.2f}")
            if 'explanation' in rec:
                print(f"   Why: {rec['explanation']}")
        
        # Get user insights
        print(f"\n🔍 Getting insights for user {user_id}...")
        insights = recommender.get_user_insights(user_id)
        
        print("📊 User Insights:")
        print(f"   Total donations: {insights.get('total_donations', 0)}")
        print(f"   Total amount: ${insights.get('total_amount', 0):,.2f}")
        print(f"   Favorite categories: {insights.get('favorite_categories', [])}")
        print(f"   Donation behavior: {insights.get('donation_behavior', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error using recommendation system: {e}")


def performance_comparison_example(db_manager: DatabaseManager):
    """Example comparing JSON vs Database performance"""
    print("\n⚡ Performance comparison example...")
    
    import time
    from src.app.comprehensive_recommender import ComprehensiveRecommendationEngine
    
    user_id = 1
    num_recommendations = 5
    
    try:
        # Test database version
        print("Testing database version...")
        db_recommender = ComprehensiveRecommendationEngineDB(db_manager)
        
        start_time = time.time()
        db_recommendations = db_recommender.get_recommendations(user_id, num_recommendations)
        db_time = time.time() - start_time
        
        print(f"✅ Database version: {db_time:.4f} seconds")
        
        # Test JSON version (if files exist)
        if os.path.exists("data/campaigns.json") and os.path.exists("data/donations.json"):
            print("Testing JSON version...")
            json_recommender = ComprehensiveRecommendationEngine(
                "data/campaigns.json",
                "data/donations.json"
            )
            
            start_time = time.time()
            json_recommendations = json_recommender.get_recommendations(str(user_id), num_recommendations)
            json_time = time.time() - start_time
            
            print(f"✅ JSON version: {json_time:.4f} seconds")
            
            # Compare
            speedup = json_time / db_time if db_time > 0 else 1
            print(f"📈 Database is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than JSON")
        else:
            print("⚠️ JSON files not found. Skipping JSON performance test.")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


def main():
    """Main example function"""
    print("🚀 Enhanced Recommendation System - Database Integration Example")
    print("=" * 70)
    
    # 1. Setup database
    db_manager = setup_database_example()
    
    # 2. Migrate data (optional)
    # migrate_json_data_example()
    
    # 3. Use recommendation system
    recommendation_system_example(db_manager)
    
    # 4. Performance comparison
    performance_comparison_example(db_manager)
    
    print("\n✨ Example completed!")
    print("\n📚 Next steps:")
    print("1. Review the DATABASE_SETUP_GUIDE.md for detailed setup instructions")
    print("2. Configure your production database settings")
    print("3. Run the migration script with your actual data")
    print("4. Integrate the database-enabled components into your application")


if __name__ == "__main__":
    main()