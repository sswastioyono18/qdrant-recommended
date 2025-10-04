# Database Integration for Enhanced Recommendation System

This guide provides a quick overview of how to use the enhanced recommendation system with database backends instead of JSON files.

## 🚀 Quick Start

### 1. Choose Your Database

**SQLite (Recommended for development)**
```python
db_config = {
    'type': 'sqlite',
    'database': 'recommendations.db'
}
```

**PostgreSQL (Recommended for production)**
```python
db_config = {
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'recommendations',
    'username': 'your_username',
    'password': 'your_password'
}
```

**MySQL**
```python
db_config = {
    'type': 'mysql',
    'host': 'localhost',
    'port': 3306,
    'database': 'recommendations',
    'username': 'your_username',
    'password': 'your_password'
}
```

### 2. Migrate Your Data

If you have existing JSON files, migrate them to the database:

```python
from src.app.migrate_to_db import migrate_json_to_database

migrate_json_to_database(
    campaigns_file="data/campaigns.json",
    donations_file="data/donations.json",
    db_config=db_config
)
```

### 3. Use the Database-Enabled System

```python
from src.app.database import DatabaseManager
from src.app.integration_guide_db import EnhancedRecommendationSystemDB

# Initialize database
db_manager = DatabaseManager(db_config)

# Initialize recommendation system
recommender = EnhancedRecommendationSystemDB(db_manager)

# Get recommendations
recommendations = recommender.get_enhanced_recommendations(
    user_id=123,
    max_recommendations=5,
    include_explanations=True
)

# Get user insights
insights = recommender.get_user_insights(user_id=123)
```

## 📁 Database-Enabled Components

All core components have database versions:

| Component | JSON Version | Database Version |
|-----------|-------------|------------------|
| Donation Analyzer | `DonationAnalyzer` | `DonationAnalyzerDB` |
| User Profiler | `AdvancedUserProfiler` | `AdvancedUserProfilerDB` |
| Campaign Filter | `SmartCampaignFilter` | `SmartCampaignFilterDB` |
| Recommendation Engine | `ComprehensiveRecommendationEngine` | `ComprehensiveRecommendationEngineDB` |
| Integration Guide | `EnhancedRecommendationSystem` | `EnhancedRecommendationSystemDB` |

## 🔧 Installation Requirements

```bash
# Core dependencies
pip install sqlite3  # Built-in with Python

# For PostgreSQL
pip install psycopg2-binary

# For MySQL
pip install mysql-connector-python
```

## 📊 Database Schema

The system creates these tables automatically:

- **campaigns**: Campaign information and metadata
- **donations**: User donation history
- **users**: User profiles and information
- **user_preferences**: User preferences and settings

## 🎯 Example Usage

Run the example script to see the database integration in action:

```bash
python example_database_usage.py
```

## 📚 Detailed Documentation

For comprehensive setup instructions, see:
- [DATABASE_SETUP_GUIDE.md](DATABASE_SETUP_GUIDE.md) - Complete setup guide
- [ENHANCED_RECOMMENDATION_SYSTEM.md](ENHANCED_RECOMMENDATION_SYSTEM.md) - Full system documentation

## 🔄 Migration from JSON

The migration process:
1. Creates database tables
2. Reads JSON files
3. Transforms and validates data
4. Inserts data into database
5. Provides migration summary

## ⚡ Performance Benefits

Database integration provides:
- **Faster queries**: Indexed database queries vs. file parsing
- **Better scalability**: Handle larger datasets efficiently
- **Data integrity**: ACID compliance and constraints
- **Concurrent access**: Multiple users can access simultaneously
- **Advanced filtering**: SQL-based filtering and aggregation

## 🛠️ Troubleshooting

**Connection Issues**:
- Verify database credentials
- Check network connectivity
- Ensure database server is running

**Migration Issues**:
- Verify JSON file format
- Check file permissions
- Ensure sufficient disk space

**Performance Issues**:
- Add database indexes
- Optimize query patterns
- Consider connection pooling

## 🔮 Next Steps

1. Set up your preferred database
2. Run the migration script
3. Update your application to use database components
4. Monitor performance and optimize as needed
5. Consider adding caching for frequently accessed data

For production deployment, see the production guide in the main documentation.