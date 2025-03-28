"""API routes for the application."""

from typing import Dict, Any
from flask import Blueprint, request, jsonify
from sqlalchemy.orm import Session
from werkzeug.exceptions import BadRequest, Unauthorized

from ..models.user import User
from ..services.auth import AuthService

# Create blueprint
api = Blueprint("api", __name__)


def get_db() -> Session:
    """Get database session."""
    # In a real app, this would get the session from Flask-SQLAlchemy
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///app.db")
    Session = sessionmaker(bind=engine)
    return Session()


@api.route("/users", methods=["POST"])
def create_user() -> Dict[str, Any]:
    """Create a new user."""
    data = request.get_json()

    if not data or not all(k in data for k in ["username", "email", "password"]):
        raise BadRequest("Missing required fields")

    db = get_db()
    auth_service = AuthService(db)

    user = auth_service.create_user(
        username=data["username"], email=data["email"], password=data["password"]
    )

    if not user:
        raise BadRequest("Username or email already exists")

    return jsonify(user.to_dict()), 201


@api.route("/auth/login", methods=["POST"])
def login() -> Dict[str, Any]:
    """Authenticate a user."""
    data = request.get_json()

    if not data or not all(k in data for k in ["username", "password"]):
        raise BadRequest("Missing username or password")

    db = get_db()
    user = db.query(User).filter_by(username=data["username"]).first()

    if not user or not user.is_active:
        raise Unauthorized("Invalid credentials")

    auth_service = AuthService(db)
    if not auth_service.verify_password(user, data["password"]):
        raise Unauthorized("Invalid credentials")

    auth_service.update_last_login(user)
    return jsonify(user.to_dict())


@api.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id: int) -> Dict[str, Any]:
    """Get user information."""
    db = get_db()
    user = db.query(User).get_or_404(user_id)
    return jsonify(user.to_dict())


@api.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id: int) -> Dict[str, Any]:
    """Update user information."""
    data = request.get_json()

    if not data:
        raise BadRequest("No data provided")

    db = get_db()
    user = db.query(User).get_or_404(user_id)

    if "email" in data:
        user.email = data["email"]

    if "is_active" in data:
        user.is_active = data["is_active"]

    db.commit()
    return jsonify(user.to_dict())
