# Database Integration Setup Guide

This guide will help you migrate your recommendation system from JSON files to a database backend.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Database Setup](#database-setup)
3. [Data Migration](#data-migration)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Dependencies

Install the required database dependencies:

```bash
# For SQLite (included in Python standard library)
# No additional installation needed

# For PostgreSQL
pip install psycopg2-binary

# For MySQL
pip install mysql-connector-python

# For general database operations
pip install sqlalchemy  # Optional, for advanced ORM features
```

### Database Software

Choose and install your preferred database:

**SQLite (Recommended for development)**
- No installation needed
- File-based database
- Perfect for testing and development

**PostgreSQL (Recommended for production)**
```bash
# macOS
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Create database
createdb recommendations_db
```

**MySQL**
```bash
# macOS
brew install mysql
brew services start mysql

# Ubuntu/Debian
sudo apt-get install mysql-server

# Create database
mysql -u root -p
CREATE DATABASE recommendations_db;
```

## Database Setup

### 1. Choose Your Database Configuration

Create a configuration based on your database choice:

**SQLite Configuration (Development)**
```python
from src.app.database import DatabaseConfig, DatabaseManager

config = DatabaseConfig(
    db_type="sqlite",
    database="recommendations.db"
)
```

**PostgreSQL Configuration (Production)**
```python
config = DatabaseConfig(
    db_type="postgresql",
    host="localhost",
    port=5432,
    database="recommendations_db",
    username="your_username",
    password="your_password"
)
```

**MySQL Configuration**
```python
config = DatabaseConfig(
    db_type="mysql",
    host="localhost",
    port=3306,
    database="recommendations_db",
    username="your_username",
    password="your_password"
)
```

### 2. Initialize Database Tables

The database tables will be created automatically when you run the migration script, but you can also create them manually:

```python
from src.app.database import DatabaseManager

db_manager = DatabaseManager(config)
db_manager.create_tables()
```

### 3. Database Schema

The system creates the following tables:

**campaigns**
- `id` (VARCHAR PRIMARY KEY)
- `title` (TEXT)
- `description` (TEXT)
- `category` (VARCHAR)
- `target_amount` (DECIMAL)
- `current_amount` (DECIMAL)
- `location` (VARCHAR)
- `urgency_level` (VARCHAR)
- `created_at` (TIMESTAMP)
- `end_date` (TIMESTAMP)
- `image_url` (TEXT)
- `organizer` (VARCHAR)
- `tags` (TEXT)

**donations**
- `id` (VARCHAR PRIMARY KEY)
- `user_id` (VARCHAR)
- `campaign_id` (VARCHAR)
- `amount` (DECIMAL)
- `donation_date` (TIMESTAMP)
- `is_anonymous` (BOOLEAN)
- `message` (TEXT)

**users**
- `id` (VARCHAR PRIMARY KEY)
- `name` (VARCHAR)
- `email` (VARCHAR)
- `created_at` (TIMESTAMP)

## Data Migration

### 1. Prepare Your JSON Files

Ensure your JSON files are in the correct format:

**campaigns.json**
```json
[
  {
    "id": "1",
    "title": "Help Build School",
    "description": "Building a school for underprivileged children",
    "category": "Education",
    "target_amount": 50000000,
    "current_amount": 15000000,
    "location": "Jakarta",
    "urgency_level": "high",
    "created_at": "2024-01-01T00:00:00",
    "end_date": "2024-12-31T23:59:59",
    "image_url": "https://example.com/image.jpg",
    "organizer": "Education Foundation",
    "tags": ["education", "children", "school"]
  }
]
```

**donations.json**
```json
[
  {
    "id": "1",
    "user_id": "1001",
    "campaign_id": "1",
    "amount": 100000,
    "donation_date": "2024-01-15T10:30:00",
    "is_anonymous": false,
    "message": "Great cause!",
    "user_name": "John Doe"
  }
]
```

### 2. Run Migration Script

Use the migration script to transfer data from JSON to database:

```bash
# Basic migration with SQLite
python src/app/migrate_to_db.py \
  --campaigns data/campaigns.json \
  --donations data/donations.json \
  --db-type sqlite \
  --db-name recommendations.db \
  --validate

# Migration with PostgreSQL
python src/app/migrate_to_db.py \
  --campaigns data/campaigns.json \
  --donations data/donations.json \
  --db-type postgresql \
  --db-host localhost \
  --db-port 5432 \
  --db-name recommendations_db \
  --db-user your_username \
  --db-password your_password \
  --validate

# Migration with MySQL
python src/app/migrate_to_db.py \
  --campaigns data/campaigns.json \
  --donations data/donations.json \
  --db-type mysql \
  --db-host localhost \
  --db-port 3306 \
  --db-name recommendations_db \
  --db-user your_username \
  --db-password your_password \
  --validate
```

### 3. Verify Migration

The migration script will show progress and results:

```
🚀 STARTING DATABASE MIGRATION
==================================================
Campaigns file: data/campaigns.json
Donations file: data/donations.json
Database type: sqlite

✅ Loaded 100 records from data/campaigns.json
✅ Loaded 1000 records from data/donations.json

📊 Migrating 100 campaigns...
✅ Campaigns migration completed: 100/100 (100.0%)

💰 Migrating 1000 donations...
👥 Creating 50 users...
✅ Donations migration completed: 1000/1000 (100.0%)
✅ Users migration completed: 50/50 (100.0%)

🔍 Validating migration...
   Campaigns in database: 100
   Donations in database: 1000
   Users in database: 50
✅ Data integrity validation passed

📋 MIGRATION SUMMARY
==================================================
Campaigns:
  Total: 100
  Migrated: 100
  Errors: 0
  Success Rate: 100.0%

Donations:
  Total: 1000
  Migrated: 1000
  Errors: 0
  Success Rate: 100.0%

Users:
  Total: 50
  Migrated: 50
  Errors: 0
  Success Rate: 100.0%

✅ No errors encountered during migration

🎉 Migration completed successfully!
```

## Configuration

### 1. Update Your Application Code

Replace the JSON-based components with database-enabled versions:

**Before (JSON-based)**
```python
from src.app.donation_analyzer import DonationAnalyzer
from src.app.advanced_profiler import AdvancedUserProfiler
from src.app.smart_filter import SmartCampaignFilter
from src.app.comprehensive_recommender import ComprehensiveRecommendationEngine

# Initialize with JSON files
analyzer = DonationAnalyzer("data/donations.json", "data/campaigns.json")
profiler = AdvancedUserProfiler("data/donations.json", "data/campaigns.json")
```

**After (Database-based)**
```python
from src.app.donation_analyzer_db import DonationAnalyzerDB
from src.app.advanced_profiler_db import AdvancedUserProfilerDB
from src.app.smart_filter_db import SmartCampaignFilterDB
from src.app.comprehensive_recommender_db import ComprehensiveRecommendationEngineDB
from src.app.database import DatabaseConfig, DatabaseManager

# Initialize with database
config = DatabaseConfig(db_type="sqlite", database="recommendations.db")
db_manager = DatabaseManager(config)

analyzer = DonationAnalyzerDB(db_manager)
profiler = AdvancedUserProfilerDB(db_manager)
```

### 2. Environment Configuration

Create environment variables for database configuration:

```bash
# .env file
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=recommendations_db
DB_USER=your_username
DB_PASSWORD=your_password
```

Load configuration from environment:

```python
import os
from src.app.database import DatabaseConfig

config = DatabaseConfig(
    db_type=os.getenv("DB_TYPE", "sqlite"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    database=os.getenv("DB_NAME", "recommendations.db"),
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
```

## Testing

### 1. Basic Functionality Test

Test the database integration:

```python
from src.app.integration_guide_db import EnhancedRecommendationSystemDB
from src.app.database import DatabaseConfig, DatabaseManager

# Initialize system
config = DatabaseConfig(db_type="sqlite", database="recommendations.db")
db_manager = DatabaseManager(config)
system = EnhancedRecommendationSystemDB(db_manager)

# Test user insights
insights = system.get_user_insights("1001")
print(f"User insights: {insights}")

# Test recommendations
recommendations = system.get_enhanced_recommendations("1001", max_recommendations=5)
print(f"Recommendations: {recommendations}")
```

### 2. Performance Test

Test system performance:

```python
# Run performance evaluation
performance = system.evaluate_system_performance()
print(f"System performance: {performance}")
```

### 3. Integration Test

Run the complete integration demo:

```python
from src.app.integration_guide_db import demonstrate_integration

# Run full demo
results = demonstrate_integration()
```

## Production Deployment

### 1. Database Optimization

**Indexing**
```sql
-- Add indexes for better performance
CREATE INDEX idx_donations_user_id ON donations(user_id);
CREATE INDEX idx_donations_campaign_id ON donations(campaign_id);
CREATE INDEX idx_donations_date ON donations(donation_date);
CREATE INDEX idx_campaigns_category ON campaigns(category);
CREATE INDEX idx_campaigns_created_at ON campaigns(created_at);
```

**Connection Pooling**
```python
from src.app.database import DatabaseManager

# Configure connection pooling
config = DatabaseConfig(
    db_type="postgresql",
    host="your-db-host",
    database="recommendations_db",
    username="your_username",
    password="your_password",
    pool_size=10,  # Add pool configuration
    max_overflow=20
)
```

### 2. Monitoring and Logging

Add monitoring for database operations:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monitor database operations
class MonitoredDatabaseManager(DatabaseManager):
    def execute_query(self, query, params=None):
        start_time = time.time()
        result = super().execute_query(query, params)
        execution_time = time.time() - start_time
        logger.info(f"Query executed in {execution_time:.3f}s: {query[:100]}...")
        return result
```

### 3. Backup Strategy

Set up regular database backups:

**PostgreSQL**
```bash
# Daily backup script
pg_dump recommendations_db > backup_$(date +%Y%m%d).sql

# Automated backup with cron
0 2 * * * pg_dump recommendations_db > /backups/backup_$(date +\%Y\%m\%d).sql
```

**MySQL**
```bash
# Daily backup script
mysqldump recommendations_db > backup_$(date +%Y%m%d).sql
```

**SQLite**
```bash
# Simple file copy
cp recommendations.db backup_$(date +%Y%m%d).db
```

## Troubleshooting

### Common Issues

**1. Connection Errors**
```
Error: could not connect to server: Connection refused
```
- Check if database server is running
- Verify host, port, username, and password
- Check firewall settings

**2. Migration Errors**
```
Error: table already exists
```
- Drop existing tables or use `--force` flag
- Check if migration was already run

**3. Performance Issues**
```
Slow query execution
```
- Add database indexes
- Optimize queries
- Use connection pooling
- Consider database tuning

**4. Memory Issues**
```
Out of memory during migration
```
- Process data in batches
- Increase available memory
- Use streaming for large datasets

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug mode
config = DatabaseConfig(
    db_type="sqlite",
    database="recommendations.db",
    debug=True  # Enable debug mode
)
```

### Data Validation

Validate your data before migration:

```python
def validate_json_data(file_path):
    """Validate JSON data structure."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    required_fields = {
        'campaigns': ['id', 'title', 'description', 'category', 'target_amount'],
        'donations': ['user_id', 'campaign_id', 'amount']
    }
    
    # Validate required fields
    for item in data:
        for field in required_fields.get('campaigns', []):
            if field not in item:
                print(f"Missing field {field} in {item}")
                return False
    
    return True

# Validate before migration
if validate_json_data('data/campaigns.json'):
    print("✅ Data validation passed")
else:
    print("❌ Data validation failed")
```

## Next Steps

After successful database integration:

1. **Update Documentation**: Update your system documentation to reflect database usage
2. **Monitor Performance**: Set up monitoring for database performance and query optimization
3. **Scale Considerations**: Plan for horizontal scaling if needed
4. **Backup Strategy**: Implement regular backup procedures
5. **Security**: Review and implement database security best practices

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review database logs for error details
3. Verify your database configuration
4. Test with a smaller dataset first
5. Consider using SQLite for initial testing

The database integration provides better performance, scalability, and data integrity compared to JSON files, making your recommendation system production-ready.