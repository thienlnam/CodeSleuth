"""Authentication service for user management."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..models.user import User


class AuthService:
    """Service for handling user authentication."""

    def __init__(self, db: Session):
        """Initialize the auth service with a database session."""
        self.db = db
        self.salt_length = 16

    def _hash_password(
        self, password: str, salt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Hash a password with a salt.

        Args:
            password: The password to hash
            salt: Optional salt to use (if None, generates a new one)

        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(self.salt_length)

        # Use PBKDF2 with SHA256
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,  # Number of iterations
        )

        return key.hex(), salt

    def create_user(self, username: str, email: str, password: str) -> Optional[User]:
        """
        Create a new user.

        Args:
            username: Username for the new user
            email: Email address for the new user
            password: Password for the new user

        Returns:
            Created User object or None if creation failed
        """
        try:
            password_hash, salt = self._hash_password(password)

            user = User(
                username=username,
                email=email,
                password_hash=f"{salt}:{password_hash}",  # Store salt with hash
            )

            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user

        except IntegrityError:
            self.db.rollback()
            return None

    def verify_password(self, user: User, password: str) -> bool:
        """
        Verify a password against a user's stored hash.

        Args:
            user: User object to verify password against
            password: Password to verify

        Returns:
            True if password is correct, False otherwise
        """
        salt, stored_hash = user.password_hash.split(":")
        computed_hash, _ = self._hash_password(password, salt)
        return secrets.compare_digest(computed_hash, stored_hash)

    def update_last_login(self, user: User) -> None:
        """
        Update the user's last login timestamp.

        Args:
            user: User to update
        """
        user.last_login = datetime.utcnow()
        self.db.commit()

    def deactivate_user(self, user: User) -> None:
        """
        Deactivate a user account.

        Args:
            user: User to deactivate
        """
        user.is_active = False
        self.db.commit()

    def reactivate_user(self, user: User) -> None:
        """
        Reactivate a user account.

        Args:
            user: User to reactivate
        """
        user.is_active = True
        self.db.commit()
