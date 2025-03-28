"""Tests for the CodeSleuth API module."""

import unittest
from unittest.mock import Mock, patch, MagicMock

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
        # Mock objects
        mock_parser = Mock()
        mock_parser_cls.return_value = mock_parser

        # Call index_repository
        self.api.index_repository()

        # Verify that the right functions were called
        mock_parser_cls.assert_called_once_with(self.config)
        mock_index_repo.assert_called_once_with(
            self.config, mock_parser, self.mock_semantic_search.semantic_index
        )


if __name__ == "__main__":
    unittest.main()
