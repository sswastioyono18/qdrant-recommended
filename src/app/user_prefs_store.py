"""
JSON-based storage for user preferences.

Provides lightweight persistence for user preference data without requiring
database tables. Useful for early-stage setups or environments where
schema changes are deferred.
"""

import os
import json
from typing import Dict, Any


class UserPreferencesStore:
    """Stores and retrieves user preferences in a JSON file."""

    def __init__(self, path: str = "data/user_preferences.json"):
        self.path = path
        # Ensure parent directory exists
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load preferences from disk, if available."""
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._data = {}
        except Exception:
            # On read error, start with empty store to avoid breaking flow
            self._data = {}

    def save(self) -> None:
        """Persist preferences to disk."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get preferences for a specific user."""
        return self._data.get(user_id, {})

    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Set preferences for a user and save to disk."""
        self._data[user_id] = preferences
        self.save()