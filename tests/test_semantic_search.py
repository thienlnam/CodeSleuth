"""Tests for the semantic search module."""

import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from codesleuth.config import CodeSleuthConfig, EmbeddingModel, IndexConfig
from codesleuth.indexing.parser import CodeChunk
from codesleuth.search.semantic_search import SemanticSearch


class TestSemanticSearch(unittest.TestCase):
    """Test cases for the semantic search module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock semantic index
        self.mock_index = Mock()

        # Create the semantic search object with the mock index
        self.search = SemanticSearch(self.mock_index)

        # Mock search results from the index
        self.mock_results = [
            (
                CodeChunk(
                    file_path="/path/to/file1.py",
                    start_line=10,
                    end_line=20,
                    code="def test_function():\n    return 'test'",
                    symbol_name="test_function",
                ),
                0.9,
            ),
            (
                CodeChunk(
                    file_path="/path/to/file2.py",
                    start_line=15,
                    end_line=25,
                    code="class TestClass:\n    def method(self):\n        pass",
                    symbol_name="TestClass",
                ),
                0.7,
            ),
        ]

    def test_search(self):
        """Test semantic search."""
        # Configure the mock index to return our test results
        self.mock_index.search.return_value = self.mock_results

        # Perform the search
        results = self.search.search(query="test query", top_k=2)

        # Check that the index search was called with the right parameters
        self.mock_index.search.assert_called_once_with("test query", top_k=2)

        # Verify the results format
        self.assertEqual(len(results), 2)

        # Check the first result
        self.assertEqual(results[0]["file_path"], "/path/to/file1.py")
        self.assertEqual(results[0]["start_line"], 10)
        self.assertEqual(results[0]["end_line"], 20)
        self.assertEqual(results[0]["symbol_name"], "test_function")
        self.assertEqual(results[0]["code"], "def test_function():\n    return 'test'")
        self.assertEqual(results[0]["similarity"], 0.9)
        self.assertEqual(results[0]["source"], "semantic")

        # Check the second result
        self.assertEqual(results[1]["file_path"], "/path/to/file2.py")
        self.assertEqual(results[1]["start_line"], 15)
        self.assertEqual(results[1]["end_line"], 25)
        self.assertEqual(results[1]["symbol_name"], "TestClass")
        self.assertEqual(
            results[1]["code"], "class TestClass:\n    def method(self):\n        pass"
        )
        self.assertEqual(results[1]["similarity"], 0.7)
        self.assertEqual(results[1]["source"], "semantic")

    def test_empty_results(self):
        """Test semantic search with empty results."""
        # Configure the mock index to return empty results
        self.mock_index.search.return_value = []

        # Perform the search
        results = self.search.search(query="nonexistent query", top_k=10)

        # Check that we got an empty list
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
