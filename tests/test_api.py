"""Tests for the CodeSleuth API module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from codesleuth.api import CodeSleuth
from codesleuth.config import CodeSleuthConfig


class TestCodeSleuthAPI(unittest.TestCase):
    """Test cases for the CodeSleuth API."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = Mock(spec=CodeSleuthConfig)
        self.config.repo_path = "/test/repo"

        # Create mocks for search modules
        self.mock_semantic_search = Mock()
        self.mock_lexical_search = Mock()

        # Patch the create functions to return our mocks
        with patch(
            "codesleuth.api.create_semantic_search"
        ) as self.mock_create_semantic:
            with patch(
                "codesleuth.api.create_lexical_search"
            ) as self.mock_create_lexical:
                # Configure mocks
                self.mock_create_semantic.return_value = self.mock_semantic_search
                self.mock_create_lexical.return_value = self.mock_lexical_search

                # Create the API instance
                self.api = CodeSleuth(self.config)

                # Verify the search modules were created with the right config
                self.mock_create_semantic.assert_called_once_with(self.config)
                self.mock_create_lexical.assert_called_once_with(self.config)

    def test_search_semantically(self):
        """Test semantic search through the API."""
        # Mock search results
        semantic_results = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": "test_function",
                "code": "def test_function():\n    return 'test'",
                "similarity": 0.9,
                "source": "semantic",
            }
        ]
        self.mock_semantic_search.search.return_value = semantic_results

        # Perform the search
        results = self.api.search_semantically("test query", top_k=5)

        # Check that the semantic search was called with the right parameters
        self.mock_semantic_search.search.assert_called_once_with("test query", top_k=5)

        # Verify the results match the mock results
        self.assertEqual(results, semantic_results)

    def test_search_lexically(self):
        """Test lexical search through the API."""
        # Mock search results
        lexical_results = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": None,
                "code": "def test_function():\n    return 'test'",
                "match_line": 10,
                "matched_text": "test_function",
                "similarity": 1.0,
                "source": "lexical",
            }
        ]
        self.mock_lexical_search.search.return_value = lexical_results

        # Perform the search
        results = self.api.search_lexically(
            "test_function",
            max_results=20,
            case_sensitive=True,
            include_pattern="*.py",
            exclude_pattern="*test*",
        )

        # Check that the lexical search was called with the right parameters
        self.mock_lexical_search.search.assert_called_once_with(
            "test_function",
            max_results=20,
            case_sensitive=True,
            include_pattern="*.py",
            exclude_pattern="*test*",
        )

        # Verify the results match the mock results
        self.assertEqual(results, lexical_results)

    def test_search_function(self):
        """Test function search through the API."""
        # Mock search results
        function_results = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": None,
                "code": "def test_function():\n    return 'test'",
                "match_line": 10,
                "matched_text": "def test_function():",
                "similarity": 1.0,
                "source": "lexical",
            }
        ]
        self.mock_lexical_search.search_function.return_value = function_results

        # Perform the search
        results = self.api.search_function("test_function", max_results=10)

        # Check that the function search was called with the right parameters
        self.mock_lexical_search.search_function.assert_called_once_with(
            "test_function", max_results=10
        )

        # Verify the results match the mock results
        self.assertEqual(results, function_results)

    def test_search_file(self):
        """Test file search through the API."""
        # Mock search results
        file_results = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 1,
                "end_line": 1,
                "symbol_name": None,
                "code": "File: file1.py",
                "similarity": 1.0,
                "source": "file_search",
            }
        ]
        self.mock_lexical_search.search_file.return_value = file_results

        # Perform the search
        results = self.api.search_file("*.py")

        # Check that the file search was called with the right parameters
        self.mock_lexical_search.search_file.assert_called_once_with("*.py")

        # Verify the results match the mock results
        self.assertEqual(results, file_results)

    def test_hybrid_search(self):
        """Test hybrid search through the API."""
        # Mock search results
        semantic_results = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": "test_function",
                "code": "def test_function():\n    return 'test'",
                "similarity": 0.9,
                "source": "semantic",
            }
        ]

        lexical_results = [
            {
                "file_path": "/test/repo/file2.py",  # Different file
                "start_line": 15,
                "end_line": 25,
                "symbol_name": None,
                "code": "class TestClass:\n    pass",
                "match_line": 15,
                "matched_text": "TestClass",
                "similarity": 1.0,
                "source": "lexical",
            }
        ]

        # Configure mocks
        self.mock_semantic_search.search.return_value = semantic_results
        self.mock_lexical_search.search.return_value = lexical_results

        # Perform the hybrid search
        results = self.api.hybrid_search("test", top_k=5, max_lexical=10)

        # Check that both search methods were called with the right parameters
        self.mock_semantic_search.search.assert_called_once_with("test", top_k=5)
        self.mock_lexical_search.search.assert_called_once_with(
            "test",
            max_results=10,
            case_sensitive=False,
            include_pattern=None,
            exclude_pattern=None,
        )

        # Verify the results - should contain both semantic and lexical results
        self.assertEqual(len(results), 2)

        # Results should be sorted by similarity (highest first)
        self.assertEqual(results[0]["source"], "lexical")  # 1.0 similarity
        self.assertEqual(results[1]["source"], "semantic")  # 0.9 similarity

    def test_hybrid_search_with_overlap(self):
        """Test hybrid search with overlapping results."""
        # Mock search results with same file/line range
        semantic_results = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": "test_function",
                "code": "def test_function():\n    return 'test'",
                "similarity": 0.9,
                "source": "semantic",
            }
        ]

        lexical_results = [
            {
                "file_path": "/test/repo/file1.py",  # Same file and range
                "start_line": 10,
                "end_line": 20,
                "symbol_name": None,
                "code": "def test_function():\n    return 'test'",
                "match_line": 10,
                "matched_text": "test_function",
                "similarity": 1.0,
                "source": "lexical",
            }
        ]

        # Configure mocks
        self.mock_semantic_search.search.return_value = semantic_results
        self.mock_lexical_search.search.return_value = lexical_results

        # Perform the hybrid search
        results = self.api.hybrid_search("test", top_k=5)

        # Verify the results - should only contain one result
        # (duplicate was filtered out)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "semantic")

    @patch("codesleuth.api.CodeParser")
    @patch("codesleuth.api.index_repository")
    def test_index_repository(self, mock_index_repo, mock_parser_cls):
        """Test repository indexing through the API."""
        # Configure mocks
        mock_parser_instance = Mock()
        mock_parser_cls.return_value = mock_parser_instance

        # Call the API method
        self.api.index_repository()

        # Verify that the parser was created with the right config
        mock_parser_cls.assert_called_once_with(self.config)

        # Verify that index_repository was called with the right parameters
        mock_index_repo.assert_called_once_with(
            self.config,
            mock_parser_instance,
            self.mock_semantic_search.semantic_index,
        )

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="line1\nline2\nline3\nline4\nline5\n",
    )
    @patch("os.path.isabs")
    @patch("os.path.join")
    def test_view_file(self, mock_join, mock_isabs, mock_open):
        """Test viewing file contents through the API."""
        # Test viewing the whole file (default behavior)
        mock_isabs.return_value = False
        mock_join.return_value = "/test/repo/file.py"

        result = self.api.view_file("file.py")
        mock_open.assert_called_once_with("/test/repo/file.py", "r", encoding="utf-8")
        self.assertEqual(result, "line1\nline2\nline3\nline4\nline5\n")

        # Reset mocks
        mock_open.reset_mock()
        mock_isabs.reset_mock()
        mock_join.reset_mock()

        # Test viewing specific line range
        mock_isabs.return_value = False
        mock_join.return_value = "/test/repo/file.py"

        result = self.api.view_file("file.py", start_line=2, end_line=4)
        mock_open.assert_called_once_with("/test/repo/file.py", "r", encoding="utf-8")
        self.assertEqual(
            result, "line1\nline2\nline3\nline4\nline5\n"
        )  # We'll get all lines due to context

        # Reset mocks
        mock_open.reset_mock()
        mock_isabs.reset_mock()
        mock_join.reset_mock()

        # Test with absolute path
        mock_isabs.return_value = True  # Set this BEFORE calling view_file

        result = self.api.view_file("/absolute/path/file.py")
        mock_open.assert_called_once_with(
            "/absolute/path/file.py", "r", encoding="utf-8"
        )
        mock_join.assert_not_called()  # Should not join paths for absolute paths

    @patch(
        "os.path.realpath"
    )  # Only mock realpath to ensure paths are consistent in tests
    def test_get_project_structure(self, mock_realpath):
        """Test getting project structure through the API."""
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock for realpath to ensure consistent paths in test
            mock_realpath.side_effect = lambda x: x

            # Update config to use temp directory
            self.config.repo_path = temp_dir

            # Create test directory structure:
            # temp_dir/
            #   ├── file1.py
            #   ├── file2.py
            #   ├── .git/           # Should be ignored
            #   │   └── config
            #   └── subdir/
            #       └── file3.py

            # Create files and directories
            os.makedirs(os.path.join(temp_dir, "subdir"))
            os.makedirs(os.path.join(temp_dir, ".git"))

            # Create test files
            for file_path in [
                os.path.join(temp_dir, "file1.py"),
                os.path.join(temp_dir, "file2.py"),
                os.path.join(temp_dir, "subdir", "file3.py"),
                os.path.join(temp_dir, ".git", "config"),
            ]:
                with open(file_path, "w") as f:
                    f.write("# Test file")

            # Get project structure
            result = self.api.get_project_structure()

            # Verify the structure
            expected = {
                "root": {
                    "file1.py": None,
                    "file2.py": None,
                    "subdir": {"file3.py": None},
                }
            }

            self.assertEqual(result, expected)

            # Test error cases

            # 1. Test with non-existent directory
            self.config.repo_path = os.path.join(temp_dir, "nonexistent")
            result = self.api.get_project_structure()
            self.assertIn("error", result)
            self.assertIn("does not exist", result["error"])

            # 2. Test with a file instead of directory
            self.config.repo_path = os.path.join(temp_dir, "file1.py")
            result = self.api.get_project_structure()
            self.assertIn("error", result)
            self.assertIn("not a directory", result["error"])

            # 3. Test with empty directory
            empty_dir = os.path.join(temp_dir, "empty")
            os.makedirs(empty_dir)
            self.config.repo_path = empty_dir
            result = self.api.get_project_structure()
            self.assertIn("error", result)
            self.assertIn("No files found", result["error"])

            # 4. Test with only hidden files
            hidden_dir = os.path.join(temp_dir, "hidden")
            os.makedirs(hidden_dir)
            with open(os.path.join(hidden_dir, ".hidden"), "w") as f:
                f.write("hidden file")
            self.config.repo_path = hidden_dir
            result = self.api.get_project_structure()
            self.assertIn("error", result)
            self.assertIn("No files found", result["error"])

    @patch("subprocess.run")
    @patch("os.path.isabs")
    @patch("os.path.join")
    def test_get_code_metadata(self, mock_join, mock_isabs, mock_run):
        """Test getting code metadata through the API."""
        # Setup mocks
        mock_isabs.return_value = False
        mock_join.return_value = "/test/repo/file.py"

        # Mock subprocess.run for functions
        function_process = Mock()
        function_process.returncode = 0
        function_process.stdout = """
        {"type":"match","data":{"line_number":10,"lines":{"text":"def function1():"},"submatches":[{"match":{"text":"def"}},{"match":{"text":"function1"}}]}}
        {"type":"match","data":{"line_number":20,"lines":{"text":"def function2(arg):"},"submatches":[{"match":{"text":"def"}},{"match":{"text":"function2"}}]}}
        """

        # Mock subprocess.run for classes
        class_process = Mock()
        class_process.returncode = 0
        class_process.stdout = """
        {"type":"match","data":{"line_number":30,"lines":{"text":"class Class1:"},"submatches":[{"match":{"text":"Class1"}}]}}
        {"type":"match","data":{"line_number":40,"lines":{"text":"class Class2(BaseClass):"},"submatches":[{"match":{"text":"Class2"}}]}}
        """

        # Configure mock_run to return different values based on the command
        def side_effect(*args, **kwargs):
            if "(function|def)" in args[0][2]:
                return function_process
            elif "class" in args[0][2]:
                return class_process
            return Mock()

        mock_run.side_effect = side_effect

        # Get code metadata
        result = self.api.get_code_metadata("file.py")

        # Verify the results
        expected = {
            "functions": ["function1", "function2"],
            "classes": ["Class1", "Class2"],
        }

        self.assertEqual(result, expected)
        self.assertEqual(mock_run.call_count, 2)  # Called for functions and classes

    @patch("codesleuth.api.CodeSleuth.search_lexically")
    def test_search_references(self, mock_search_lexically):
        """Test searching for references through the API."""
        # Setup mocks for regular references
        mock_search_lexically.return_value = [
            {
                "file_path": "/test/repo/file1.py",
                "start_line": 10,
                "end_line": 20,
                "symbol_name": None,
                "code": "def test_symbol():\n    return True",
                "match_line": 10,
                "matched_text": "test_symbol",
                "similarity": 1.0,
                "source": "lexical",
            }
        ]

        # Test regular references search
        result = self.api.search_references("test_symbol", definition=False)

        # Verify that search_lexically was called with the right parameters
        mock_search_lexically.assert_called_once_with(
            "\\btest_symbol\\b", max_results=20, case_sensitive=True
        )

        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["matched_text"], "test_symbol")

        # Reset mock
        mock_search_lexically.reset_mock()

        # Setup mock return values for definition search
        pattern_results = [
            # Function pattern result
            [
                {
                    "file_path": "/test/repo/file1.py",
                    "start_line": 10,
                    "end_line": 15,
                    "code": "def test_symbol():\n    pass",
                    "match_line": 10,
                    "matched_text": "def test_symbol()",
                    "similarity": 1.0,
                    "source": "lexical",
                }
            ],
            # Class pattern result
            [
                {
                    "file_path": "/test/repo/file2.py",
                    "start_line": 20,
                    "end_line": 25,
                    "code": "class test_symbol:\n    pass",
                    "match_line": 20,
                    "matched_text": "class test_symbol",
                    "similarity": 1.0,
                    "source": "lexical",
                }
            ],
            # Variable pattern results
            [],  # Empty for JS/TS
            [],  # Empty for other languages
        ]

        mock_search_lexically.side_effect = pattern_results

        # Test definition search
        result = self.api.search_references("test_symbol", definition=True)

        # Verify each pattern was searched
        self.assertEqual(mock_search_lexically.call_count, 4)

        # Check results were combined and deduplicated
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["file_path"], "/test/repo/file1.py")
        self.assertEqual(result[1]["file_path"], "/test/repo/file2.py")


if __name__ == "__main__":
    unittest.main()
