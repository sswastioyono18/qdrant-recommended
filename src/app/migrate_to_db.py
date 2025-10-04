"""
DATABASE MIGRATION SCRIPT

This script migrates data from JSON files to the database.
It handles the migration of projects, donations, and users data.

Usage:
    python migrate_to_db.py --projects data/projects.json --donations data/donations.json

Features:
- Creates database tables if they don't exist
- Migrates projects, donations, and users
- Handles data validation and error reporting
- Provides progress tracking
- Supports different database backends (SQLite, PostgreSQL, MySQL)
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Use package-relative imports to avoid run-as-module issues
from .database import DatabaseConfig, DatabaseManager
from .models import Project, Donation, User, ProjectRepository, DonationRepository, UserRepository


class DataMigrator:
    """Handles migration of data from JSON files to database."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the migrator with database manager."""
        self.db_manager = db_manager
        self.project_repo = ProjectRepository(db_manager)
        self.donation_repo = DonationRepository(db_manager)
        self.user_repo = UserRepository(db_manager)
        
        # Migration statistics
        self.stats = {
            "projects": {"total": 0, "migrated": 0, "errors": 0},
            "donations": {"total": 0, "migrated": 0, "errors": 0},
            "users": {"total": 0, "migrated": 0, "errors": 0}
        }
        
        self.errors = []
    
    def load_json_file(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Load and parse JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data)} records from {file_path}")
                return data
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def migrate_projects(self, projects_data: List[Dict[str, Any]]) -> bool:
        """Migrate projects data to database."""
        print(f"\n📊 Migrating {len(projects_data)} projects...")
        self.stats["projects"]["total"] = len(projects_data)
        
        for i, project_data in enumerate(projects_data, 1):
            try:
                # Validate required fields
                # Support flexible schemas (e.g., campaigns.json)
                # Require minimal fields and derive missing ones
                required_fields = ["id", "title"]
                for field in required_fields:
                    if field not in project_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Create project object
                project = Project(
                    id=str(project_data["id"]),
                    title=project_data["title"],
                    # Fallback to html_description if description is missing
                    description=project_data.get("description") or project_data.get("html_description", ""),
                    # Use category string if available, else category_id
                    category=str(project_data.get("category", project_data.get("category_id", ""))),
                    # Use 0.0 if target_amount is not provided
                    target_amount=float(project_data.get("target_amount", 0.0)),
                    current_amount=float(project_data.get("current_amount", 0)),
                    # Map location from country if available
                    location=project_data.get("location", project_data.get("country", "")),
                    urgency_level=project_data.get("urgency_level", "medium"),
                    created_at=project_data.get("created_at", datetime.now().isoformat()),
                    end_date=project_data.get("end_date"),
                    image_url=project_data.get("image_url", ""),
                    organizer=project_data.get("organizer", ""),
                    tags=",".join(project_data.get("tags", [])) if isinstance(project_data.get("tags"), list) else project_data.get("tags", "")
                )
                
                # Save to database
                saved_project = self.project_repo.create(project)
                if saved_project:
                    self.stats["projects"]["migrated"] += 1
                    if i % 10 == 0:
                        print(f"   Migrated {i}/{len(projects_data)} projects...")
                else:
                    self.stats["projects"]["errors"] += 1
                    self.errors.append(f"Failed to save project {project_data['id']}")
                
            except Exception as e:
                self.stats["projects"]["errors"] += 1
                error_msg = f"Error migrating project {project_data.get('id', 'unknown')}: {str(e)}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        success_rate = (self.stats["projects"]["migrated"] / self.stats["projects"]["total"]) * 100
        print(f"✅ Projects migration completed: {self.stats['projects']['migrated']}/{self.stats['projects']['total']} ({success_rate:.1f}%)")
        return self.stats["projects"]["errors"] == 0
    
    def migrate_donations(self, donations_data: List[Dict[str, Any]]) -> bool:
        """Migrate donations data to database."""
        # Support donations.json that may be a dict keyed by user_id
        flattened_donations: List[Dict[str, Any]] = []
        unique_users: Dict[str, Dict[str, Any]] = {}

        if isinstance(donations_data, dict):
            # donations_data like { "1001": [ {campaign_id, amount, ts}, ... ], ... }
            for user_id, user_donations in donations_data.items():
                str_user_id = str(user_id)
                # Register user
                if str_user_id not in unique_users:
                    unique_users[str_user_id] = {
                        "id": str_user_id,
                        "name": f"User {str_user_id}",
                        "email": f"user{str_user_id}@example.com",
                        "created_at": datetime.now().isoformat(),
                    }
                # Flatten each donation
                for entry in user_donations:
                    flattened_donations.append({
                        "user_id": str_user_id,
                        # Map campaign_id to project_id
                        "project_id": str(entry.get("campaign_id", "")),
                        "amount": entry.get("amount", 0),
                        # Map ts to donation_date
                        "donation_date": entry.get("ts", datetime.now().isoformat()),
                        "is_anonymous": False,
                        "message": "",
                    })
        else:
            # donations_data is already a list of donation dicts
            flattened_donations = donations_data
            # Extract unique users from list
            for entry in flattened_donations:
                user_id = str(entry.get("user_id", ""))
                if user_id and user_id not in unique_users:
                    unique_users[user_id] = {
                        "id": user_id,
                        "name": entry.get("user_name", f"User {user_id}"),
                        "email": entry.get("email", f"user{user_id}@example.com"),
                        "created_at": datetime.now().isoformat(),
                    }

        print(f"\n💰 Migrating {len(flattened_donations)} donations...")
        self.stats["donations"]["total"] = len(flattened_donations)
        
        # Migrate users first
        print(f"👥 Creating {len(unique_users)} users...")
        self.stats["users"]["total"] = len(unique_users)
        
        for user_data in unique_users.values():
            try:
                user = User(
                    id=user_data["id"],
                    name=user_data["name"],
                    email=user_data["email"],
                    created_at=user_data["created_at"]
                )
                
                saved_user = self.user_repo.create(user)
                if saved_user:
                    self.stats["users"]["migrated"] += 1
                else:
                    self.stats["users"]["errors"] += 1
                    self.errors.append(f"Failed to save user {user_data['id']}")
                    
            except Exception as e:
                self.stats["users"]["errors"] += 1
                error_msg = f"Error migrating user {user_data.get('id', 'unknown')}: {str(e)}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        # Now migrate donations
        for i, donation_data in enumerate(flattened_donations, 1):
            try:
                # Validate required fields
                required_fields = ["user_id", "project_id", "amount"]
                for field in required_fields:
                    if field not in donation_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Create donation object
                donation = Donation(
                    id=str(donation_data.get("id", f"donation_{i}")),
                    user_id=str(donation_data["user_id"]),
                    project_id=str(donation_data["project_id"]),
                    amount=float(donation_data["amount"]),
                    donation_date=donation_data.get("donation_date", datetime.now().isoformat()),
                    is_anonymous=donation_data.get("is_anonymous", False),
                    message=donation_data.get("message", "")
                )
                
                # Save to database
                saved_donation = self.donation_repo.create(donation)
                if saved_donation:
                    self.stats["donations"]["migrated"] += 1
                    if i % 50 == 0:
                        print(f"   Migrated {i}/{len(donations_data)} donations...")
                else:
                    self.stats["donations"]["errors"] += 1
                    self.errors.append(f"Failed to save donation {donation_data.get('id', i)}")
                
            except Exception as e:
                self.stats["donations"]["errors"] += 1
                error_msg = f"Error migrating donation {donation_data.get('id', i)}: {str(e)}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        success_rate = (self.stats["donations"]["migrated"] / self.stats["donations"]["total"]) * 100 if self.stats["donations"]["total"] > 0 else 100
        print(f"✅ Donations migration completed: {self.stats['donations']['migrated']}/{self.stats['donations']['total']} ({success_rate:.1f}%)")
        
        user_success_rate = (self.stats["users"]["migrated"] / self.stats["users"]["total"]) * 100 if self.stats["users"]["total"] > 0 else 100
        print(f"✅ Users migration completed: {self.stats['users']['migrated']}/{self.stats['users']['total']} ({user_success_rate:.1f}%)")
        
        return self.stats["donations"]["errors"] == 0 and self.stats["users"]["errors"] == 0
    
    def validate_migration(self) -> bool:
        """Validate the migration by checking data integrity."""
        print(f"\n🔍 Validating migration...")
        
        # Check if all projects exist
        all_projects = self.project_repo.get_all()
        print(f"   Projects in database: {len(all_projects)}")
        
        # Check if all donations exist
        all_donations = self.donation_repo.get_all()
        print(f"   Donations in database: {len(all_donations)}")
        
        # Check if all users exist
        all_users = self.user_repo.get_all()
        print(f"   Users in database: {len(all_users)}")
        
        # Validate referential integrity
        integrity_errors = 0
        
        for donation in all_donations[:100]:  # Check first 100 donations
            # Check if project exists
            project = self.project_repo.get_by_id(donation.project_id)
            if not project:
                print(f"❌ Donation {donation.id} references non-existent project {donation.project_id}")
                integrity_errors += 1
            
            # Check if user exists
            user = self.user_repo.get_by_id(donation.user_id)
            if not user:
                print(f"❌ Donation {donation.id} references non-existent user {donation.user_id}")
                integrity_errors += 1
        
        if integrity_errors == 0:
            print("✅ Data integrity validation passed")
            return True
        else:
            print(f"❌ Found {integrity_errors} integrity errors")
            return False
    
    def print_migration_summary(self):
        """Print a summary of the migration results."""
        print(f"\n📋 MIGRATION SUMMARY")
        print("=" * 50)
        
        for entity_type, stats in self.stats.items():
            if stats["total"] > 0:
                success_rate = (stats["migrated"] / stats["total"]) * 100
                print(f"{entity_type.title()}:")
                print(f"  Total: {stats['total']}")
                print(f"  Migrated: {stats['migrated']}")
                print(f"  Errors: {stats['errors']}")
                print(f"  Success Rate: {success_rate:.1f}%")
                print()
        
        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        else:
            print("✅ No errors encountered during migration")


def setup_database(db_manager: DatabaseManager, init_config: Optional[DatabaseConfig] = None, skip_create: bool = False):
    """Set up database tables.
    - If `skip_create` is True, skip creating tables.
    - If `init_config` is provided, create tables in the separate init database.
    - Otherwise, create tables in the main database referenced by `db_manager`.
    """
    if skip_create:
        print("⏭️ Skipping table creation per configuration")
        return True

    print("🔧 Setting up database tables...")
    try:
        if init_config:
            print("📦 Creating tables in separate init database...")
            init_manager = DatabaseManager(init_config)
            init_manager.connect()
            init_manager.create_tables()
            init_manager.disconnect()
        else:
            db_manager.create_tables()
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return False


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate data from JSON files to database")
    parser.add_argument("--projects", required=True, help="Path to projects JSON file")
    parser.add_argument("--donations", required=True, help="Path to donations JSON file")
    parser.add_argument("--db-type", default="sqlite", choices=["sqlite", "postgresql", "mysql"], 
                       help="Database type (default: sqlite)")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, help="Database port")
    parser.add_argument("--db-name", default="recommendations.db", help="Database name")
    parser.add_argument("--sqlite-path", default="data/recommendation.db", help="SQLite database file path when using sqlite")
    parser.add_argument("--db-user", help="Database username")
    parser.add_argument("--db-password", help="Database password")
    parser.add_argument("--validate", action="store_true", help="Validate migration after completion")
    # Table creation controls for separate init DB
    parser.add_argument("--skip-create-tables", action="store_true", help="Skip table creation entirely")
    parser.add_argument("--init-separate", action="store_true", help="Create tables in a separate initialization database")
    parser.add_argument("--init-db-type", choices=["sqlite", "postgresql", "mysql"], help="Init DB type (default: same as --db-type)")
    parser.add_argument("--init-db-host", help="Init DB host")
    parser.add_argument("--init-db-port", type=int, help="Init DB port")
    parser.add_argument("--init-db-name", help="Init DB name")
    parser.add_argument("--init-db-user", help="Init DB username")
    parser.add_argument("--init-db-password", help="Init DB password")
    parser.add_argument("--init-sqlite-path", help="Init SQLite file path when using sqlite")
    
    args = parser.parse_args()
    
    print("🚀 STARTING DATABASE MIGRATION")
    print("=" * 50)
    print(f"Projects file: {args.projects}")
    print(f"Donations file: {args.donations}")
    print(f"Database type: {args.db_type}")
    print()
    
    # Configure database
    config = DatabaseConfig(
        db_type=args.db_type,
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        username=args.db_user,
        password=args.db_password,
        sqlite_path=args.sqlite_path if args.db_type == "sqlite" else None
    )
    
    # Initialize database manager
    try:
        db_manager = DatabaseManager(config)
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False
    
    # Build optional separate init config
    init_config = None
    if args.init_separate:
        # Determine defaults for ports if not provided
        init_db_type = args.init_db_type or args.db_type
        default_port = 5432 if init_db_type == "postgresql" else (3306 if init_db_type == "mysql" else None)
        init_port = args.init_db_port if args.init_db_port is not None else (args.db_port if args.db_port is not None else default_port)

        init_config = DatabaseConfig(
            db_type=init_db_type,
            host=args.init_db_host or args.db_host,
            port=init_port,
            database=args.init_db_name or args.db_name,
            username=args.init_db_user or (args.db_user or ""),
            password=args.init_db_password or (args.db_password or ""),
            sqlite_path=args.init_sqlite_path or "data/recommendation.db",
        )

    # Set up database tables
    if not setup_database(db_manager, init_config=init_config, skip_create=args.skip_create_tables):
        return False
    
    # Initialize migrator
    migrator = DataMigrator(db_manager)
    
    # Load JSON files
    projects_data = migrator.load_json_file(args.projects)
    donations_data = migrator.load_json_file(args.donations)
    
    if not projects_data or not donations_data:
        print("❌ Failed to load required JSON files")
        return False
    
    # Perform migration
    success = True
    
    # Migrate projects first (donations reference projects)
    if not migrator.migrate_projects(projects_data):
        success = False
    
    # Migrate donations and users
    if not migrator.migrate_donations(donations_data):
        success = False
    
    # Validate migration if requested
    if args.validate:
        if not migrator.validate_migration():
            success = False
    
    # Print summary
    migrator.print_migration_summary()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("\n📝 Next steps:")
        print("1. Update your application to use the database-enabled components")
        print("2. Test the recommendation system with database data")
        print("3. Monitor performance and optimize queries as needed")
    else:
        print("\n⚠️ Migration completed with errors. Please review the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)