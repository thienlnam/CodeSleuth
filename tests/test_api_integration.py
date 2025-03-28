"""Integration tests for CodeSleuth API with a sample repository."""

import os
import shutil
import unittest
import logging
from pathlib import Path
from typing import Optional

from codesleuth import CodeSleuth, CodeSleuthConfig, IndexConfig, EmbeddingModel
from codesleuth.config import ParserConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestCodeSleuthIntegration(unittest.TestCase):
    """Integration tests for CodeSleuth with a sample repository."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        try:
            # Get the path to the sample repository
            cls.sample_repo_path = Path(__file__).parent / "fixtures" / "sample_repo"
            logger.debug(f"Sample repo path: {cls.sample_repo_path}")

            # Create a temporary directory for the index
            cls.index_path = cls.sample_repo_path / "index"
            cls.index_path.mkdir(exist_ok=True)
            logger.debug(f"Index path: {cls.index_path}")

            # Create configuration
            parser_config = ParserConfig(
                chunk_size=100,
                chunk_overlap=20,
                ignore_patterns=[r"\.git", r"__pycache__", r".*\.pyc", r"node_modules"],
                respect_gitignore=True,
            )

            # Try to use a simpler model first
            index_config = IndexConfig(
                model_name=EmbeddingModel.CODEBERT,
                index_path=str(cls.index_path),
                batch_size=8,
                use_gpu=False,
                hnsw_m=16,
                hnsw_ef_construction=200,
                hnsw_ef_search=50,
            )

            cls.config = CodeSleuthConfig(
                repo_path=str(cls.sample_repo_path),
                parser=parser_config,
                index=index_config,
            )

            # Initialize CodeSleuth
            logger.debug("Initializing CodeSleuth...")
            cls.codesleuth = CodeSleuth(cls.config)
            logger.debug("CodeSleuth initialized successfully")

            # Check if semantic search is available
            if cls.codesleuth.is_semantic_search_available():
                logger.debug("Semantic search is available, indexing repository...")
                cls.codesleuth.index_repository()
                logger.debug("Repository indexed successfully")
            else:
                logger.warning("Semantic search is not available, skipping indexing")

        except Exception as e:
            logger.error(f"Error in setUpClass: {e}", exc_info=True)
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        try:
            # Remove the index directory
            if cls.index_path.exists():
                shutil.rmtree(cls.index_path)
                logger.debug("Index directory cleaned up")
        except Exception as e:
            logger.error(f"Error in tearDownClass: {e}", exc_info=True)

    def test_semantic_search(self):
        """Test semantic search functionality."""
        if not self.codesleuth.is_semantic_search_available():
            self.skipTest("Semantic search is not available")

        try:
            # Test searching for authentication-related code
            results = self.codesleuth.search_semantically(
                "user authentication and password hashing", top_k=5
            )

            self.assertGreater(
                len(results), 0, "Should find authentication-related code"
            )

            # Check that we found the auth service
            auth_service_found = any(
                "auth.py" in result["file_path"] for result in results
            )
            self.assertTrue(
                auth_service_found, "Should find the auth service implementation"
            )
        except Exception as e:
            logger.error(f"Error in test_semantic_search: {e}", exc_info=True)
            raise

    def test_lexical_search(self):
        """Test lexical search functionality."""
        try:
            # Test searching for specific function names
            results = self.codesleuth.search_lexically(
                "create_user", case_sensitive=True
            )

            self.assertGreater(len(results), 0, "Should find create_user function")

            # Check that we found it in the auth service
            auth_service_found = any(
                "auth.py" in result["file_path"] for result in results
            )
            self.assertTrue(
                auth_service_found, "Should find create_user in auth service"
            )
        except Exception as e:
            logger.error(f"Error in test_lexical_search: {e}", exc_info=True)
            raise

    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        try:
            results = self.codesleuth.hybrid_search(
                "user authentication and password hashing", top_k=5
            )

            self.assertGreater(
                len(results), 0, "Should find authentication-related code"
            )

            # Check that we got both semantic and lexical results
            sources = {result["source"] for result in results}
            self.assertGreater(
                len(sources), 1, "Should have both semantic and lexical results"
            )
        except Exception as e:
            logger.error(f"Error in test_hybrid_search: {e}", exc_info=True)
            raise

    def test_search_function_definitions(self):
        """Test searching for function definitions."""
        try:
            results = self.codesleuth.search_function_definitions(
                "create_user", max_results=5
            )

            self.assertGreater(
                len(results), 0, "Should find create_user function definition"
            )

            # Check that we found it in the auth service
            auth_service_found = any(
                "auth.py" in result["file_path"] for result in results
            )
            self.assertTrue(
                auth_service_found, "Should find create_user in auth service"
            )
        except Exception as e:
            logger.error(
                f"Error in test_search_function_definitions: {e}", exc_info=True
            )
            raise

    def test_search_references(self):
        """Test searching for references to functions."""
        try:
            results = self.codesleuth.search_references("create_user", max_results=5)

            self.assertGreater(len(results), 0, "Should find references to create_user")

            # Check that we found it in the API routes
            api_routes_found = any(
                "routes.py" in result["file_path"] for result in results
            )
            self.assertTrue(
                api_routes_found, "Should find create_user reference in API routes"
            )
        except Exception as e:
            logger.error(f"Error in test_search_references: {e}", exc_info=True)
            raise

    def test_get_project_structure(self):
        """Test getting project structure."""
        try:
            structure = self.codesleuth.get_project_structure()

            self.assertIn("root", structure, "Should have root key")
            self.assertIn("src", structure["root"], "Should have src directory")
            self.assertIn(
                "frontend", structure["root"], "Should have frontend directory"
            )

            # Check for specific files
            src_files = structure["root"]["src"]
            self.assertIn("main.py", src_files, "Should have main.py")
            self.assertIn("api", src_files, "Should have api directory")
            self.assertIn("models", src_files, "Should have models directory")
        except Exception as e:
            logger.error(f"Error in test_get_project_structure: {e}", exc_info=True)
            raise

    def test_get_code_metadata(self):
        """Test getting code metadata across different programming languages."""
        try:
            # Test Python-style code (auth.py)
            python_metadata = self.codesleuth.get_code_metadata("src/services/auth.py")

            self.assertIn("functions", python_metadata, "Should have functions key")
            self.assertIn("classes", python_metadata, "Should have classes key")

            # Check Python functions and classes
            create_user_functions = [
                f for f in python_metadata["functions"] if "def create_user" in f
            ]
            self.assertTrue(
                len(create_user_functions) > 0,
                "Should find at least one create_user function definition",
            )
            create_user_func = create_user_functions[0]
            self.assertIn(
                "def create_user(self, username: str, email: str, password: str) -> Optional[User]",
                create_user_func,
            )
            self.assertIn("Create a new user", create_user_func)
            self.assertIn("Args:", create_user_func)
            self.assertIn("Returns:", create_user_func)
            self.assertIn(
                "password_hash, salt = self._hash_password(password)", create_user_func
            )

            # Check for AuthService class
            auth_service_classes = [
                c for c in python_metadata["classes"] if "class AuthService" in c
            ]
            self.assertTrue(
                len(auth_service_classes) > 0,
                "Should find at least one AuthService class definition",
            )
            auth_service_class = auth_service_classes[0]
            self.assertIn("class AuthService:", auth_service_class)
            self.assertIn(
                "Service for handling user authentication", auth_service_class
            )
            self.assertIn("def __init__(self, db: Session)", auth_service_class)

            # Test TypeScript/JavaScript-style code
            ts_metadata = self.codesleuth.get_code_metadata(
                "src/services/multi_lang_test.ts"
            )

            # Check TypeScript functions
            process_user_functions = [
                f for f in ts_metadata["functions"] if "function processUser" in f
            ]
            self.assertTrue(
                len(process_user_functions) > 0,
                "Should find at least one processUser function definition",
            )
            process_user_func = process_user_functions[0]
            self.assertIn("function processUser(user: User)", process_user_func)
            self.assertIn("return user.process()", process_user_func)

            # Check async function
            fetch_user_functions = [
                f
                for f in ts_metadata["functions"]
                if "async function fetchUserData" in f
            ]
            self.assertTrue(
                len(fetch_user_functions) > 0,
                "Should find at least one fetchUserData async function definition",
            )
            fetch_user_func = fetch_user_functions[0]
            self.assertIn("async function fetchUserData(id: string)", fetch_user_func)
            self.assertIn("return await api.getUser(id)", fetch_user_func)

            # Check class methods
            create_user_methods = [
                f for f in ts_metadata["functions"] if "createUser" in f
            ]
            self.assertTrue(
                len(create_user_methods) > 0,
                "Should find at least one createUser method",
            )
            create_user_method = create_user_methods[0]
            self.assertIn("createUser(username: string)", create_user_method)
            self.assertIn("Promise<User>", create_user_method)

            # Check TypeScript classes
            user_manager_classes = [
                c for c in ts_metadata["classes"] if "class UserManager" in c
            ]
            self.assertTrue(
                len(user_manager_classes) > 0,
                "Should find at least one UserManager class definition",
            )
            user_manager_class = user_manager_classes[0]
            self.assertIn("class UserManager", user_manager_class)
            self.assertIn("private users: Map<string, User>", user_manager_class)
            self.assertIn("public async createUser", user_manager_class)

            # Check abstract class
            abstract_classes = [
                c for c in ts_metadata["classes"] if "abstract class" in c
            ]
            self.assertTrue(
                len(abstract_classes) > 0,
                "Should find at least one abstract class definition",
            )
            abstract_class = abstract_classes[0]
            self.assertIn("abstract class", abstract_class)
            self.assertIn("protected abstract init()", abstract_class)

            # Test Rust-style code
            rust_metadata = self.codesleuth.get_code_metadata(
                "src/services/user_service.rs"
            )

            # Check Rust functions
            process_user_functions = [
                f for f in rust_metadata["functions"] if "fn process_user" in f
            ]
            self.assertTrue(
                len(process_user_functions) > 0,
                "Should find at least one process_user function definition",
            )
            process_user_func = process_user_functions[0]
            self.assertIn("pub fn process_user(user: &User) -> bool", process_user_func)
            self.assertIn("user.is_active()", process_user_func)

            # Check async function
            fetch_user_functions = [
                f for f in rust_metadata["functions"] if "fn fetch_user_data" in f
            ]
            self.assertTrue(
                len(fetch_user_functions) > 0,
                "Should find at least one fetch_user_data function definition",
            )
            fetch_user_func = fetch_user_functions[0]
            self.assertIn(
                "async fn fetch_user_data(id: &str) -> Result<User, String>",
                fetch_user_func,
            )
            self.assertIn("tokio::time::sleep", fetch_user_func)

            # Check Rust structs (treated as classes)
            user_structs = [c for c in rust_metadata["classes"] if "struct User" in c]
            self.assertTrue(
                len(user_structs) > 0, "Should find at least one User struct definition"
            )
            user_struct = user_structs[0]
            self.assertIn("pub struct User", user_struct)
            self.assertIn("id: String", user_struct)
            self.assertIn("username: String", user_struct)

        except Exception as e:
            logger.error(f"Error in test_get_code_metadata: {e}", exc_info=True)
            raise

    def test_view_file(self):
        """Test viewing file contents."""
        try:
            # Test viewing auth service file
            content = self.codesleuth.view_file(
                "src/services/auth.py", start_line=1, end_line=20
            )

            self.assertIsInstance(content, str, "Should return string content")
            self.assertGreater(len(content), 0, "Should have content")
            self.assertIn(
                "class AuthService", content, "Should contain AuthService class"
            )
        except Exception as e:
            logger.error(f"Error in test_view_file: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    unittest.main()
