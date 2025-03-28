"""Main application entry point."""

import os
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models.user import Base
from .api.routes import api


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configure database
    db_path = os.path.join(os.path.dirname(__file__), "..", "app.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Initialize database
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
    Base.metadata.create_all(engine)

    # Register blueprints
    app.register_blueprint(api, url_prefix="/api")

    return app


def main():
    """Run the application."""
    app = create_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
