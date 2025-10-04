"""
Database connection and configuration utilities for the recommendation system.
Supports PostgreSQL, MySQL, and SQLite databases.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager
import sqlite3
import json
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration class"""
    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "recommendation_db"
    username: str = ""
    password: str = ""
    sqlite_path: str = "data/recommendation.db"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables"""
        return cls(
            db_type=os.getenv('DB_TYPE', 'sqlite'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'recommendation_db'),
            username=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            sqlite_path=os.getenv('SQLITE_PATH', 'data/recommendation.db')
        )

class DatabaseManager:
    """Database manager for handling different database types"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            if self.config.db_type == "sqlite":
                self.connection = sqlite3.connect(self.config.sqlite_path)
                self.connection.row_factory = sqlite3.Row
                try:
                    logger.info(f"Connecting to SQLite at path: {self.config.sqlite_path}")
                except Exception:
                    pass
                
            elif self.config.db_type == "postgresql":
                if not POSTGRES_AVAILABLE:
                    raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
                
                self.connection = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password
                )
                try:
                    logger.info(f"Connecting to PostgreSQL host={self.config.host} db={self.config.database} port={self.config.port}")
                except Exception:
                    pass
                
            elif self.config.db_type == "mysql":
                if not MYSQL_AVAILABLE:
                    raise ImportError("mysql-connector-python not installed. Install with: pip install mysql-connector-python")
                
                self.connection = mysql.connector.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password
                )
                try:
                    logger.info(f"Connecting to MySQL host={self.config.host} db={self.config.database} port={self.config.port}")
                except Exception:
                    pass
                
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
                
            logger.info(f"Connected to {self.config.db_type} database")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic cleanup"""
        if not self.connection:
            self.connect()
        
        # Use appropriate cursor types per backend
        if self.config.db_type == "postgresql":
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        elif self.config.db_type == "mysql":
            # Buffered dictionary cursor prevents "Unread result found" when issuing subsequent queries
            cursor = self.connection.cursor(buffered=True, dictionary=True)
        else:
            cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        with self.get_cursor() as cursor:
            # Log the exact SQL and parameters being executed for transparency/debugging
            try:
                logger.info(
                    f"Executing SQL [{self.config.db_type}] -> {query.strip()} | params={params or ()}"
                )
            except Exception:
                # Avoid logging-related failures from breaking query execution
                pass
            cursor.execute(query, params or ())
            
            if self.config.db_type == "sqlite":
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            
            elif self.config.db_type == "postgresql":
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            
            elif self.config.db_type == "mysql":
                # Rows already dicts; ensure list conversion
                return list(cursor.fetchall())
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_cursor() as cursor:
            try:
                logger.info(
                    f"Executing SQL UPDATE [{self.config.db_type}] -> {query.strip()} | params={params or ()}"
                )
            except Exception:
                pass
            cursor.execute(query, params or ())
            return cursor.rowcount

    def get_table_columns(self, table_name: str) -> List[str]:
        """Return column names for a table across supported databases."""
        with self.get_cursor() as cursor:
            try:
                if self.config.db_type == "sqlite":
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    rows = cursor.fetchall()
                    return [row[1] if isinstance(row, tuple) else row[1] for row in rows]
                elif self.config.db_type == "postgresql":
                    cursor.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table_name,)
                    )
                    return [r[0] if not isinstance(r, dict) else r.get('column_name') for r in cursor.fetchall()]
                elif self.config.db_type == "mysql":
                    cursor = self.connection.cursor(buffered=True)
                    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
                    rows = cursor.fetchall()
                    # MySQL returns tuples like (Field, Type, Null, Key, Default, Extra)
                    return [row[0] for row in rows]
                else:
                    return []
            except Exception as e:
                logger.error(f"Failed to get columns for table '{table_name}': {e}")
                return []

    def get_column_type(self, table_name: str, column_name: str) -> Optional[str]:
        """Return the database-native column type for the given table.column."""
        try:
            if self.config.db_type == "sqlite":
                with self.get_cursor() as cursor:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    rows = cursor.fetchall()
                    for row in rows:
                        # SQLite: (cid, name, type, notnull, dflt_value, pk)
                        if isinstance(row, tuple):
                            name = row[1]
                            col_type = row[2]
                        else:
                            name = row.get('name')
                            col_type = row.get('type')
                        if name == column_name:
                            return col_type
                return None
            elif self.config.db_type == "postgresql":
                with self.get_cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT data_type, character_maximum_length
                        FROM information_schema.columns
                        WHERE table_name = %s AND column_name = %s
                        """,
                        (table_name, column_name)
                    )
                    rows = cursor.fetchall()
                    if rows:
                        r = rows[0]
                        if isinstance(r, dict):
                            dt = r.get('data_type')
                            l = r.get('character_maximum_length')
                        else:
                            dt = r[0]
                            l = r[1]
                        if dt and l:
                            return f"{dt.upper()}({l})"
                        return dt.upper() if dt else None
                return None
            elif self.config.db_type == "mysql":
                # MySQL returns exact type strings including length and signedness
                with self.get_cursor() as cursor:
                    cursor = self.connection.cursor(buffered=True)
                    cursor.execute(f"SHOW COLUMNS FROM `{table_name}` LIKE '{column_name}'")
                    row = cursor.fetchone()
                    if row:
                        # (Field, Type, Null, Key, Default, Extra)
                        return row[1]
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get type for {table_name}.{column_name}: {e}")
            return None
    
    def create_tables(self):
        """Create database tables for the recommendation system"""
        # Allow skipping the user_preferences table when using JSON store
        use_json_prefs = os.getenv('USER_PREFERENCES_JSON', '0') == '1'
        # Determine auto-increment primary key syntax per database
        if self.config.db_type == 'sqlite':
            pk_auto = 'INTEGER PRIMARY KEY AUTOINCREMENT'
        elif self.config.db_type == 'postgresql':
            pk_auto = 'SERIAL PRIMARY KEY'
        elif self.config.db_type == 'mysql':
            pk_auto = 'INT AUTO_INCREMENT PRIMARY KEY'
        else:
            pk_auto = 'INTEGER PRIMARY KEY'
        # Build user_preferences table SQL dynamically to inject PK syntax
        # Align FK type to users.id; detect existing type if table exists, else default to schema's VARCHAR(50).
        user_fk_detected = self.get_column_type('users', 'id')
        user_fk_type = user_fk_detected or 'VARCHAR(50)'
        user_preferences_sql = f'''
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id {pk_auto},
                        user_id {user_fk_type} NOT NULL,
                        category VARCHAR(100),
                        preference_score DECIMAL(3,2) DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                '''
        tables = {
            'projects': '''
                CREATE TABLE IF NOT EXISTS projects (
                    id VARCHAR(50) PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    category VARCHAR(100),
                    target_amount DECIMAL(15,2),
                    current_amount DECIMAL(15,2) DEFAULT 0,
                    location VARCHAR(200),
                    urgency_level VARCHAR(20),
                    project_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active'
                )
            ''',
            'donations': '''
                CREATE TABLE IF NOT EXISTS donations (
                    id VARCHAR(50) PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    projects_id VARCHAR(50) NOT NULL,
                    amount DECIMAL(15,2) NOT NULL,
                    donation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_anonymous BOOLEAN DEFAULT FALSE,
                    payment_method VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (projects_id) REFERENCES projects(id)
                )
            ''',
            'users': '''
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(50) PRIMARY KEY,
                    email VARCHAR(255) UNIQUE,
                    full_name VARCHAR(255),
                    location VARCHAR(200),
                    age_group VARCHAR(20),
                    income_level VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active'
                )
            ''',
            # Only include the user_preferences table if JSON store is disabled
            **({
                'user_preferences': user_preferences_sql
            } if not use_json_prefs else {})
        }

        for table_name, create_sql in tables.items():
            try:
                self.execute_update(create_sql)
                logger.info(f"Table '{table_name}' created successfully")
            except Exception as e:
                logger.error(f"Failed to create table '{table_name}': {e}")
                raise

# Global database manager instance
db_manager: Optional[DatabaseManager] = None

def get_db_manager() -> DatabaseManager:
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        config = DatabaseConfig.from_env()
        db_manager = DatabaseManager(config)
    return db_manager

def init_database():
    """Initialize database connection and create tables.

    Supports separating table creation from the main application database via environment variables.
    Env controls:
      - `DB_INIT_ENABLED` (default: "1"): if "0", skip table creation entirely.
      - `DB_INIT_SEPARATE` (default: "0"): if "1", create tables in a separate init database.
      - `INIT_DB_TYPE`, `INIT_DB_HOST`, `INIT_DB_PORT`, `INIT_DB_NAME`, `INIT_DB_USER`, `INIT_DB_PASSWORD`, `INIT_SQLITE_PATH`:
         override the separate init database connection.
    """
    manager = get_db_manager()
    manager.connect()

    # Determine if table creation should occur and where
    init_enabled = os.getenv('DB_INIT_ENABLED', '1')
    init_separate = os.getenv('DB_INIT_SEPARATE', '0')

    if init_enabled == '0':
        logger.info("DB_INIT_ENABLED=0, skipping table creation")
        return manager

    # If using a separate init database, construct and create there
    if init_separate == '1':
        # Build separate init config from env, with sensible fallbacks
        init_db_type = os.getenv('INIT_DB_TYPE') or os.getenv('DB_TYPE', 'sqlite')
        init_host = os.getenv('INIT_DB_HOST', os.getenv('DB_HOST', 'localhost'))
        init_port = int(os.getenv('INIT_DB_PORT', os.getenv('DB_PORT', '5432')))
        init_name = os.getenv('INIT_DB_NAME', os.getenv('DB_NAME', 'recommendation_db'))
        init_user = os.getenv('INIT_DB_USER', os.getenv('DB_USER', ''))
        init_password = os.getenv('INIT_DB_PASSWORD', os.getenv('DB_PASSWORD', ''))
        init_sqlite_path = os.getenv('INIT_SQLITE_PATH', os.getenv('SQLITE_PATH', 'data/recommendation.db'))

        init_config = DatabaseConfig(
            db_type=init_db_type,
            host=init_host,
            port=init_port,
            database=init_name,
            username=init_user,
            password=init_password,
            sqlite_path=init_sqlite_path,
        )

        logger.info(
            f"Creating tables in separate init DB (type={init_config.db_type}, "
            f"sqlite_path={init_config.sqlite_path if init_config.db_type=='sqlite' else init_config.database})"
        )

        init_manager = DatabaseManager(init_config)
        init_manager.connect()
        init_manager.create_tables()
        init_manager.disconnect()
    else:
        # Default: create tables in the main application database
        logger.info("Creating tables in main application database")
        manager.create_tables()

    return manager