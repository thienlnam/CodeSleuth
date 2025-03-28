"""Tests for the lexical search module."""

import os
import unittest
from unittest.mock import Mock, patch, call

from codesleuth.config import CodeSleuthConfig
from codesleuth.search.lexical_search import LexicalSearch


class TestLexicalSearch(unittest.TestCase):
    """Test cases for the lexical search module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = Mock(spec=CodeSleuthConfig)
        self.config.repo_path = "/test/repo"

        # Create the lexical search object
        self.search = LexicalSearch(self.config)

    @patch("codesleuth.search.lexical_search.subprocess.run")
    def test_search(self, mock_run):
        """Test lexical search with ripgrep."""
        # Configure the mock subprocess.run to return test data
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = """
        {"type":"begin","data":{"path":{"text":"file1.py"}}}
        {"type":"match","data":{"path":{"text":"file1.py"},"line_number":15,"lines":{"text":"def test_function():"},"submatches":[{"match":{"text":"test_function"}}]}}
        {"type":"context","data":{"path":{"text":"file1.py"},"line_number":16,"lines":{"text":"    # Test comment"}}}
        {"type":"context","data":{"path":{"text":"file1.py"},"line_number":17,"lines":{"text":"    return 'test'"}}}
        {"type":"end","data":{}}
        {"type":"begin","data":{"path":{"text":"file2.py"}}}
        {"type":"match","data":{"path":{"text":"file2.py"},"line_number":25,"lines":{"text":"class TestClass:"},"submatches":[{"match":{"text":"TestClass"}}]}}
        {"type":"context","data":{"path":{"text":"file2.py"},"line_number":26,"lines":{"text":"    def method(self):"}}}
        {"type":"context","data":{"path":{"text":"file2.py"},"line_number":27,"lines":{"text":"        pass"}}}
        {"type":"end","data":{}}
        """
        mock_run.return_value = mock_process

        # Perform the search
        results = self.search.search(query="test", max_results=10)

        # Check that ripgrep was called with the right parameters
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(kwargs["text"], True)
        self.assertEqual(kwargs["capture_output"], True)
        self.assertEqual(kwargs["check"], False)

        # Verify the results
        self.assertEqual(len(results), 2)

        # Check the first result
        self.assertEqual(results[0]["file_path"], "/test/repo/file1.py")
        self.assertEqual(results[0]["match_line"], 15)
        self.assertEqual(results[0]["matched_text"], "test_function")
        self.assertEqual(results[0]["source"], "lexical")

        # Check the second result
        self.assertEqual(results[1]["file_path"], "/test/repo/file2.py")
        self.assertEqual(results[1]["match_line"], 25)
        self.assertEqual(results[1]["matched_text"], "TestClass")
        self.assertEqual(results[1]["source"], "lexical")

    @patch("codesleuth.search.lexical_search.subprocess.run")
    def test_search_with_options(self, mock_run):
        """Test lexical search with various options."""
        # Configure the mock subprocess.run to return empty data
        mock_process = Mock()
        mock_process.returncode = 1  # No matches
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        # Perform the search with options
        self.search.search(
            query="test",
            max_results=20,
            case_sensitive=True,
            include_pattern="*.py",
            exclude_pattern="*test*",
        )

        # Check that ripgrep was called with the right parameters
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("--max-count", cmd)
        self.assertEqual(cmd[cmd.index("--max-count") + 1], "20")

        # Case sensitive: -i should not be present
        self.assertNotIn("-i", cmd)

        # Include pattern
        self.assertIn("--glob", cmd)
        self.assertEqual(cmd[cmd.index("--glob") + 1], "*.py")

        # Exclude pattern
        self.assertIn("--glob", cmd, msg="--glob should appear at least twice")
        exclude_index = (
            len(cmd) - 1 - cmd[::-1].index("--glob")
        )  # Find the last occurrence
        self.assertEqual(cmd[exclude_index + 1], "!*test*")

    @patch("codesleuth.search.lexical_search.subprocess.run")
    def test_search_function(self, mock_run):
        """Test function search."""
        # Configure the mock subprocess.run to return test data
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = """
        {"type":"begin","data":{"path":{"text":"file1.py"}}}
        {"type":"match","data":{"path":{"text":"file1.py"},"line_number":15,"lines":{"text":"def test_function():"},"submatches":[{"match":{"text":"test_function"}}]}}
        {"type":"end","data":{}}
        """
        mock_run.return_value = mock_process

        # Perform the function search
        results = self.search.search_function("test_function")

        # Check that ripgrep was called with the right pattern
        args, kwargs = mock_run.call_args
        cmd = args[0]
        pattern_index = cmd.index("(function|def|class)\\s+test_function\\b")
        self.assertTrue(pattern_index > 0)

        # Verify the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["matched_text"], "test_function")

    @patch("codesleuth.search.lexical_search.subprocess.run")
    def test_search_file(self, mock_run):
        """Test file search."""
        # Configure the mock subprocess.run to return test data
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "/test/repo/file1.py\n/test/repo/file2.py"
        mock_run.return_value = mock_process

        # Perform the file search
        results = self.search.search_file("*.py")

        # Check that find was called with the right parameters
        expected_cmd = ["find", "/test/repo", "-type", "f", "-name", "*.py"]
        mock_run.assert_called_once_with(
            expected_cmd, text=True, capture_output=True, check=True
        )

        # Verify the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["file_path"], "/test/repo/file1.py")
        self.assertEqual(results[0]["source"], "file_search")
        self.assertEqual(results[1]["file_path"], "/test/repo/file2.py")
        self.assertEqual(results[1]["source"], "file_search")


if __name__ == "__main__":
    unittest.main()
