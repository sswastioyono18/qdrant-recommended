"""
Data models and repository classes for the recommendation system.
Provides abstraction layer for database operations.
Aligned to the simplified schema in `database.py` and the
`migrate_to_db.py` migration script.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os
from .database import get_db_manager

logger = logging.getLogger(__name__)

@dataclass
class Project:
    """Project data model aligned with migration script and schema"""
    id: str
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    target_amount: Optional[float] = None
    current_amount: Optional[float] = None
    location: Optional[str] = None
    urgency_level: Optional[str] = None
    project_type: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: Optional[str] = None
    # Extra fields from JSON that are not persisted
    end_date: Optional[str] = None
    image_url: Optional[str] = None
    organizer: Optional[str] = None
    tags: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        return cls(**data)

@dataclass
class Donation:
    """Donation data model aligned with migration script and schema"""
    id: str
    project_id: str  # maps to donations.projects_id
    user_id: str     # maps to donations.user_id
    amount: float
    donation_date: Optional[str] = None
    is_anonymous: bool = False
    payment_method: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Donation':
        return cls(**data)

@dataclass
class User:
    """User data model aligned with schema"""
    id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    location: Optional[str] = None
    age_group: Optional[str] = None
    income_level: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(**data)

class ProjectRepository:
    """Repository for project (campaign) data operations"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        # Detect actual column names to support varying schemas
        try:
            self._project_cols = set(self.db_manager.get_table_columns('projects'))
        except Exception:
            self._project_cols = set()

        # Helper to create select expressions with optional fallback and alias
        def expr(primary: str, alias: str = None) -> str:
            a = alias or primary
            if primary in self._project_cols:
                return primary if a == primary else f"{primary} AS {a}"
            return f"NULL AS {a}"

        def expr_fallback(primary: str, fallback: str, alias: str) -> str:
            if primary and primary in self._project_cols:
                return primary if alias == primary else f"{primary} AS {alias}"
            if fallback and fallback in self._project_cols:
                return f"{fallback} AS {alias}"
            return f"NULL AS {alias}"

        # Title can be stored as 'title' or 'name'
        title_expr = 'title' if 'title' in self._project_cols else ('name AS title' if 'name' in self._project_cols else 'NULL AS title')
        # Category may be absent or represented as foreign key id columns
        category_expr = (
            'category' if 'category' in self._project_cols else (
                'categories_id AS category' if 'categories_id' in self._project_cols else (
                    'project_categories_id AS category' if 'project_categories_id' in self._project_cols else 'NULL AS category'
                )
            )
        )
        # Type may be 'project_type' or generic 'type'
        project_type_expr = 'project_type' if 'project_type' in self._project_cols else (
            'type AS project_type' if 'type' in self._project_cols else 'NULL AS project_type'
        )
        # created_at/updated_at often appear as 'created'/'updated' in MySQL schemas
        created_at_expr = 'created_at' if 'created_at' in self._project_cols else (
            'created AS created_at' if 'created' in self._project_cols else 'NULL AS created_at'
        )
        updated_at_expr = 'updated_at' if 'updated_at' in self._project_cols else (
            'updated AS updated_at' if 'updated' in self._project_cols else 'NULL AS updated_at'
        )
        # status may be 'status' or 'state' or absent
        status_expr = 'status' if 'status' in self._project_cols else (
            'state AS status' if 'state' in self._project_cols else 'NULL AS status'
        )

        # Build the select clause once and reuse
        self._select_clause = \
            (
                "id, "
                + title_expr + ", "
                + expr('description') + ", "
                + category_expr + ", "
                + expr('target_amount') + ", "
                + expr('current_amount') + ", "
                + expr('location') + ", "
                + expr('urgency_level') + ", "
                + project_type_expr + ", "
                + created_at_expr + ", "
                + updated_at_expr + ", "
                + status_expr
            )
    
    def _ph(self) -> str:
        """Parameter placeholder based on DB type"""
        return '?' if self.db_manager.config.db_type == 'sqlite' else '%s'
    
    def get_all(self) -> List[Project]:
        """Get all projects from database"""
        query = f"""
        SELECT {self._select_clause}
        FROM projects
        ORDER BY created_at DESC
        LIMIT 1000
        """
        
        try:
            results = self.db_manager.execute_query(query)
            projects = []
            for row in results:
                project = Project(
                    id=str(row.get('id')),
                    title=row.get('title'),
                    description=row.get('description'),
                    category=row.get('category'),
                    target_amount=float(row.get('target_amount') or 0),
                    current_amount=float(row.get('current_amount') or 0),
                    location=row.get('location'),
                    urgency_level=row.get('urgency_level'),
                    project_type=row.get('project_type'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at'),
                    status=row.get('status')
                )
                projects.append(project)
            return projects
        except Exception as e:
            logging.error(f"Error fetching projects: {e}")
            return []
    
    def get_by_id(self, project_id: int) -> Optional[Project]:
        """Get project by ID"""
        query = f"""
        SELECT {self._select_clause}
        FROM projects
        WHERE id = {self._ph()}
        """

        try:
            results = self.db_manager.execute_query(query, (project_id,))
            if results:
                row = results[0]
                return Project(
                    id=str(row.get('id')),
                    title=row.get('title'),
                    description=row.get('description'),
                    category=row.get('category'),
                    target_amount=float(row.get('target_amount') or 0),
                    current_amount=float(row.get('current_amount') or 0),
                    location=row.get('location'),
                    urgency_level=row.get('urgency_level'),
                    project_type=row.get('project_type'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at'),
                    status=row.get('status')
                )
            return None
        except Exception as e:
            logging.error(f"Error fetching project {project_id}: {e}")
            return None
    
    def get_by_category(self, category_id: int) -> List[Project]:
        """Get projects by category (if category column exists)"""
        # Note: Since we only have id and title from the schema provided,
        # this method returns empty list. Update when full schema is available.
        query = f"""
        SELECT {self._select_clause}
        FROM projects
        WHERE id > 0
        ORDER BY created_at DESC
        LIMIT 100
        """
        
        try:
            results = self.db_manager.execute_query(query)
            projects = []
            for row in results:
                project = Project(
                    id=str(row.get('id')),
                    title=row.get('title'),
                    description=row.get('description'),
                    category=row.get('category'),
                    target_amount=float(row.get('target_amount') or 0),
                    current_amount=float(row.get('current_amount') or 0),
                    location=row.get('location'),
                    urgency_level=row.get('urgency_level'),
                    project_type=row.get('project_type'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at'),
                    status=row.get('status')
                )
                projects.append(project)
            return projects
        except Exception as e:
            logging.error(f"Error fetching projects by category {category_id}: {e}")
            return []

    def create(self, project: Project) -> bool:
        """Insert a project into database"""
        cols = [
            'id', 'title', 'description', 'category', 'target_amount', 'current_amount',
            'location', 'urgency_level', 'project_type', 'created_at', 'status'
        ]
        ph = ', '.join([self._ph()] * len(cols))
        query = f"INSERT INTO projects ({', '.join(cols)}) VALUES ({ph})"
        params = (
            project.id, project.title, project.description, project.category,
            project.target_amount, project.current_amount, project.location,
            project.urgency_level, project.project_type, project.created_at,
            project.status or 'active'
        )
        try:
            affected = self.db_manager.execute_update(query, params)
            return affected > 0
        except Exception as e:
            logging.error(f"Error inserting project {project.id}: {e}")
            return False

class DonationRepository:
    """Repository for donation data operations"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        # Allow overriding the donations table (e.g., 'kbcore.donations') via env
        self.table_name = os.getenv('DB_DONATIONS_TABLE', 'donations')
        # Detect actual column names to support legacy schemas
        try:
            cols = set(self.db_manager.get_table_columns(self.table_name))
        except Exception:
            cols = set()
        # Default to canonical columns
        self.col_user_id = 'user_id'
        self.col_project_id = 'projects_id'
        # Additional fields that vary across schemas
        self.col_donation_date = 'donation_date'
        self.col_created_at = 'created_at'
        self.col_payment_method = 'payment_method'
        self.col_status = 'status'
        # Legacy support: some guides use 'campaign_id' instead of 'projects_id'
        if 'campaign_id' in cols:
            self.col_project_id = 'campaign_id'
        # Support schemas using singular 'project_id'
        if 'project_id' in cols:
            self.col_project_id = 'project_id'
        # Some data sources misname user field
        if 'users_id' in cols:
            self.col_user_id = 'users_id'
        elif 'userid' in cols:
            # Some schemas use 'userid' (without underscore)
            self.col_user_id = 'userid'
        # Optional env override to force specific user id column (e.g., 'user_id' or 'users_id')
        override_user_col = os.getenv('DB_DONATIONS_USER_ID_COLUMN')
        if override_user_col:
            if override_user_col in cols:
                self.col_user_id = override_user_col
                logger.info(f"Using overridden donations user id column: {self.col_user_id}")
            else:
                logger.warning(
                    f"DB_DONATIONS_USER_ID_COLUMN='{override_user_col}' not found in donations columns; "
                    f"detected '{self.col_user_id}' will be used."
                )
        # Finished status override: allow configuring the value used for a "finished" donation
        # Defaults to 1 (commonly used for completed in numeric status schemas)
        finished_env = os.getenv('DB_DONATIONS_STATUS_FINISHED')
        if finished_env is None:
            # Default to numeric 1 to match common schemas
            self.finished_status = 1
        else:
            try:
                # Try interpreting env as int for numeric status columns
                self.finished_status = int(finished_env)
            except Exception:
                # Fallback to raw string (for schemas using text statuses)
                self.finished_status = finished_env
        # Map varying timestamp/status/payment fields to canonical aliases
        # Donation date/timestamp column variants
        if 'donated_at' in cols:
            self.col_donation_date = 'donated_at'
        elif 'donation_date' not in cols and 'created' in cols:
            self.col_donation_date = 'created'
        if 'created_at' not in cols and 'created' in cols:
            self.col_created_at = 'created'
        elif 'created_at' not in cols and 'donated_at' in cols:
            # Some schemas only have donated_at; use it as created_at fallback
            self.col_created_at = 'donated_at'
        # Payment method column variants
        if 'payment_method' not in cols and 'payment_methods_id' in cols:
            self.col_payment_method = 'payment_methods_id'
        elif 'payment_method' not in cols and 'payment_type' in cols:
            self.col_payment_method = 'payment_type'
        # Status column variants
        if 'status' not in cols and 'donation_statuses_id' in cols:
            self.col_status = 'donation_statuses_id'
        elif 'status' not in cols and 'donation_status' in cols:
            self.col_status = 'donation_status'
        elif 'status' not in cols and 'state' in cols:
            self.col_status = 'state'

        # Build select expressions with safe fallbacks and aliases
        # Amount variants and fallback
        if 'amount' in cols:
            self._amount_expr = 'amount'
        elif 'donation_amount' in cols:
            self._amount_expr = 'donation_amount AS amount'
        elif 'total' in cols:
            self._amount_expr = 'total AS amount'
        else:
            self._amount_expr = '0 AS amount'

        # is_anonymous variants and fallback to 0 if absent
        if 'is_anonymous' in cols:
            self._is_anonymous_expr = 'is_anonymous'
        elif 'anonymous' in cols:
            self._is_anonymous_expr = 'anonymous AS is_anonymous'
        elif 'is_private' in cols:
            self._is_anonymous_expr = 'is_private AS is_anonymous'
        else:
            self._is_anonymous_expr = '0 AS is_anonymous'
    
    def _ph(self) -> str:
        return '?' if self.db_manager.config.db_type == 'sqlite' else '%s'
    
    def get_all(self) -> List[Donation]:
        """Get all donations from database (limited for performance)"""
        query = f"""
        SELECT id, {self.col_project_id} AS projects_id, {self.col_user_id} AS user_id, {self._amount_expr}, {self._is_anonymous_expr}, {self.col_donation_date} AS donation_date,
               {self.col_status} AS status, {self.col_created_at} AS created_at, {self.col_payment_method} AS payment_method
        FROM {self.table_name}
        ORDER BY donation_date DESC
        LIMIT 10000
        """
        
        try:
            results = self.db_manager.execute_query(query)
            donations = []
            for row in results:
                donation = Donation(
                    id=str(row.get('id')),
                    project_id=str(row.get('projects_id')),
                    user_id=str(row.get('user_id')),
                    amount=float(row.get('amount') or 0),
                    is_anonymous=bool(row.get('is_anonymous')),
                    donation_date=row.get('donation_date'),
                    status=row.get('status'),
                    created_at=row.get('created_at'),
                    payment_method=row.get('payment_method')
                )
                donations.append(donation)
            return donations
        except Exception as e:
            logging.error(f"Error fetching donations: {e}")
            return []

    def get_by_id(self, donation_id: int) -> Optional[Donation]:
        """Get donation by ID"""
        query = f"""
        SELECT id, {self.col_project_id} AS projects_id, {self.col_user_id} AS user_id, {self._amount_expr}, {self._is_anonymous_expr}, {self.col_donation_date} AS donation_date,
               {self.col_status} AS status, {self.col_created_at} AS created_at, {self.col_payment_method} AS payment_method
        FROM {self.table_name}
        WHERE id = {self._ph()}
        """
        
        try:
            results = self.db_manager.execute_query(query, (donation_id,))
            if results:
                row = results[0]
                return Donation(
                    id=str(row.get('id')),
                    project_id=str(row.get('projects_id')),
                    user_id=str(row.get('user_id')),
                    amount=float(row.get('amount') or 0),
                    is_anonymous=bool(row.get('is_anonymous')),
                    donation_date=row.get('donation_date'),
                    status=row.get('status'),
                    created_at=row.get('created_at'),
                    payment_method=row.get('payment_method')
                )
            return None
        except Exception as e:
            logging.error(f"Error fetching donation {donation_id}: {e}")
            return None

    def get_by_user(self, user_id: int) -> List[Donation]:
        """Get donations by user ID"""
        query = f"""
        SELECT id, {self.col_project_id} AS projects_id, {self.col_user_id} AS user_id, {self._amount_expr}, {self._is_anonymous_expr}, {self.col_donation_date} AS donation_date,
               {self.col_status} AS status, {self.col_created_at} AS created_at, {self.col_payment_method} AS payment_method
        FROM {self.table_name}
        WHERE {self.col_user_id} = {self._ph()}
        ORDER BY donation_date DESC
        LIMIT 100
        """

        try:
            results = self.db_manager.execute_query(query, (user_id,))
            # print('results are: ', results)
            # print('query results: ', results)
            donations = []
            for row in results:
                donation = Donation(
                    id=str(row.get('id')),
                    project_id=str(row.get('projects_id')),
                    user_id=str(row.get('user_id')),
                    amount=float(row.get('amount') or 0),
                    is_anonymous=bool(row.get('is_anonymous')),
                    donation_date=row.get('donation_date'),
                    status=row.get('status'),
                    created_at=row.get('created_at'),
                    payment_method=row.get('payment_method')
                )
                donations.append(donation)
            return donations
        except Exception as e:
            logging.error(f"Error fetching donations for user {user_id}: {e}")
            return []

    def get_by_user_id(self, user_id: str) -> List[Donation]:
        uid = int(str(user_id).strip())
        donations = self.get_by_user(uid)
        out = list(donations)
        print("INSIDE len:", len(out), "id:", id(out))
        return out

    def get_by_project_id(self, project_id: str) -> List[Donation]:
        """Get donations filtered by project ID (maps to donations.projects_id)."""
        query = f"""
        SELECT id, {self.col_project_id} AS projects_id, {self.col_user_id} AS user_id, {self._amount_expr}, {self._is_anonymous_expr}, {self.col_donation_date} AS donation_date,
               {self.col_status} AS status, {self.col_created_at} AS created_at, {self.col_payment_method} AS payment_method
        FROM {self.table_name}
        WHERE {self.col_project_id} = {self._ph()}
        ORDER BY donation_date DESC
        LIMIT 1000
        """

        try:
            results = self.db_manager.execute_query(query, (project_id,))
            donations = []
            for row in results:
                donation = Donation(
                    id=str(row.get('id')),
                    project_id=str(row.get('projects_id')),
                    user_id=str(row.get('user_id')),
                    amount=float(row.get('amount') or 0),
                    is_anonymous=bool(row.get('is_anonymous')),
                    donation_date=row.get('donation_date'),
                    status=row.get('status'),
                    created_at=row.get('created_at'),
                    payment_method=row.get('payment_method')
                )
                donations.append(donation)
            return donations
        except Exception as e:
            logging.error(f"Error fetching donations for project {project_id}: {e}")
            return []

    def get_by_campaign_id(self, campaign_id: str) -> List[Donation]:
        """Compatibility alias for code referencing 'campaign' terminology."""
        return self.get_by_project_id(campaign_id)

    def create(self, donation: Donation) -> bool:
        """Insert a donation into database"""
        cols = [
            'id', self.col_user_id, self.col_project_id, 'amount', 'donation_date', 'is_anonymous',
            'payment_method', 'status', 'created_at'
        ]
        ph = ', '.join([self._ph()] * len(cols))
        query = f"INSERT INTO {self.table_name} ({', '.join(cols)}) VALUES ({ph})"
        params = (
            donation.id,
            donation.user_id,
            donation.project_id,
            donation.amount,
            donation.donation_date,
            int(bool(donation.is_anonymous)),
            donation.payment_method,
            donation.status or 'completed',
            donation.created_at
        )
        try:
            affected = self.db_manager.execute_update(query, params)
            return affected > 0
        except Exception as e:
            logging.error(f"Error inserting donation {donation.id}: {e}")
            return False

    def update_status(self, donation_id: str, new_status: Any) -> bool:
        """Update the status of a donation by its ID.

        Accepts either numeric codes (e.g., 1, 4) or text values
        depending on the schema. Uses parameterized queries.
        """
        try:
            query = f"UPDATE {self.table_name} SET {self.col_status} = {self._ph()} WHERE id = {self._ph()}"
            params = (new_status, donation_id)
            affected = self.db_manager.execute_update(query, params)
            return affected > 0
        except Exception as e:
            logging.error(f"Error updating donation status for {donation_id}: {e}")
            return False

    def mark_finished(self, donation_id: str) -> bool:
        """Convenience method to mark a donation as finished/completed."""
        return self.update_status(donation_id, self.finished_status)

    def mark_all_user_donations_finished(self, user_id: str) -> int:
        """Mark all donations for a user as finished; returns affected row count.

        This performs a bulk update. Use with caution in production.
        """
        try:
            query = (
                f"UPDATE {self.table_name} SET {self.col_status} = {self._ph()} "
                f"WHERE {self.col_user_id} = {self._ph()}"
            )
            params = (self.finished_status, user_id)
            affected = self.db_manager.execute_update(query, params)
            return int(affected or 0)
        except Exception as e:
            logging.error(f"Error marking donations finished for user {user_id}: {e}")
            return 0

class UserRepository:
    """Repository for user data operations"""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def _ph(self) -> str:
        return '?' if self.db_manager.config.db_type == 'sqlite' else '%s'

    def get_all(self) -> List[User]:
        """Get all users from database (limited for performance)"""
        query = """
        SELECT id, email, full_name, location, age_group, income_level,
               created_at, updated_at, status
        FROM users
        ORDER BY created_at DESC
        LIMIT 10000
        """

        try:
            results = self.db_manager.execute_query(query)
            users = []
            for row in results:
                user = User(
                    id=str(row.get('id')),
                    email=row.get('email'),
                    full_name=row.get('full_name'),
                    location=row.get('location'),
                    age_group=row.get('age_group'),
                    income_level=row.get('income_level'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at'),
                    status=row.get('status')
                )
                users.append(user)
            return users
        except Exception as e:
            logging.error(f"Error fetching users: {e}")
            return []

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        query = f"""
        SELECT id, email, full_name, location, age_group, income_level,
               created_at, updated_at, status
        FROM users
        WHERE id = {self._ph()}
        """

        try:
            results = self.db_manager.execute_query(query, (user_id,))
            if results:
                row = results[0]
                return User(
                    id=str(row.get('id')),
                    email=row.get('email'),
                    full_name=row.get('full_name'),
                    location=row.get('location'),
                    age_group=row.get('age_group'),
                    income_level=row.get('income_level'),
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at'),
                    status=row.get('status')
                )
            return None
        except Exception as e:
            logging.error(f"Error fetching user {user_id}: {e}")
            return None

# Repository factory: prefer fresh instances per request/task to avoid shared mutable state
def make_repositories():
    dbm = get_db_manager()
    return (
        ProjectRepository(dbm),
        DonationRepository(dbm),
        UserRepository(dbm),
    )