# Sample Web Application

This is a sample web application used for testing CodeSleuth's code search capabilities.

## Project Structure

-   `src/` - Main application source code
    -   `api/` - API endpoints and routes
    -   `models/` - Database models
    -   `services/` - Business logic
    -   `utils/` - Utility functions
-   `tests/` - Test files
-   `frontend/` - Frontend application
    -   `src/` - Frontend source code
    -   `components/` - React components
    -   `utils/` - Frontend utilities

## Setup

1. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python src/main.py
    ```

## Testing

Run tests with:

```bash
pytest
```
