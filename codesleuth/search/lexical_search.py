"""
Lexical search module for CodeSleuth.

This module provides lexical code search capabilities using ripgrep (rg)
for efficient and precise pattern matching.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from loguru import logger

from ..config import CodeSleuthConfig


@dataclass
class GrepResult:
    """Result from ripgrep search."""

    file_path: str
    line_number: int
    content: str
    matched_text: str


class LexicalSearch:
    """Lexical code search using ripgrep."""

    def __init__(self, config: CodeSleuthConfig):
        """
        Initialize the lexical search.

        Args:
            config: CodeSleuth configuration
        """
        self.config = config
        self.repo_path = config.repo_path

    def search(
        self,
        query: str,
        max_results: int = 50,
        case_sensitive: bool = False,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for code snippets matching the query pattern.

        Args:
            query: Query string (regex pattern supported)
            max_results: Maximum number of results to return
            case_sensitive: Whether the search should be case sensitive
            include_pattern: Glob pattern for files to include
            exclude_pattern: Glob pattern for files to exclude

        Returns:
            List of dictionaries with code snippets and metadata
        """
        logger.debug(
            f"Lexical search for: {query} (max_results={max_results}, "
            f"case_sensitive={case_sensitive}, include={include_pattern}, exclude={exclude_pattern})"
        )

        # Build ripgrep command
        cmd = ["rg", "--json"]

        # Add options
        if not case_sensitive:
            cmd.append("-i")

        # Set max results
        cmd.extend(["--max-count", str(max_results)])

        # Add include/exclude patterns
        if include_pattern:
            cmd.extend(["--glob", include_pattern])
        if exclude_pattern:
            cmd.extend(["--glob", f"!{exclude_pattern}"])

        # Add context for better code snippets
        cmd.extend(["--context", "5"])

        # Add query and path
        cmd.append(query)
        cmd.append(str(self.repo_path))

        # Run ripgrep
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, text=True, capture_output=True, check=False)

            if result.returncode not in [0, 1]:  # 0=matches found, 1=no matches
                logger.error(f"ripgrep error: {result.stderr}")
                return []

            # Parse results
            return self._parse_ripgrep_output(result.stdout)

        except Exception as e:
            logger.error(f"Failed to run ripgrep: {e}")
            return []

    def _parse_ripgrep_output(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse ripgrep JSON output.

        Args:
            output: ripgrep output in JSON format

        Returns:
            List of formatted search results
        """
        results = []
        current_match = None
        context_lines = {}

        # Process each line of JSON output
        for line in output.splitlines():
            if not line or not line.strip():
                continue

            try:
                # Use json.loads instead of eval for safety
                data = json.loads(line.strip())
                match_type = data.get("type")

                if match_type == "begin":
                    # Start of a new match
                    path = data.get("data", {}).get("path", {}).get("text", "")
                    abs_path = path
                    if not os.path.isabs(path):
                        abs_path = os.path.join(str(self.repo_path), path)

                    current_match = {
                        "file_path": abs_path,
                        "lines": [],
                        "line_number": None,
                        "matched_text": None,
                        "context_before": [],
                        "context_after": [],
                    }
                    context_lines = {}

                elif match_type == "match" and current_match:
                    # A matching line
                    match_data = data.get("data", {})
                    line_number = match_data.get("line_number", 0)
                    text = match_data.get("lines", {}).get("text", "")

                    current_match["line_number"] = line_number
                    current_match["matched_text"] = text.strip()

                    # Extract submatches if available
                    submatches = match_data.get("submatches", [])
                    if submatches:
                        matched_text = submatches[0].get("match", {}).get("text", "")
                        current_match["matched_text"] = matched_text

                elif match_type == "context" and current_match:
                    # Context lines
                    context_data = data.get("data", {})
                    line_number = context_data.get("line_number", 0)
                    text = context_data.get("lines", {}).get("text", "")

                    context_lines[line_number] = text.strip()

                elif match_type == "end" and current_match:
                    # End of a match, organize context lines
                    match_line = current_match["line_number"]

                    for line_num, text in sorted(context_lines.items()):
                        if line_num < match_line:
                            current_match["context_before"].append((line_num, text))
                        elif line_num > match_line:
                            current_match["context_after"].append((line_num, text))

                    # Format the final result
                    start_line = match_line - len(current_match["context_before"])
                    end_line = match_line + len(current_match["context_after"])

                    # Build code snippet with context
                    code_lines = []
                    for line_num, text in sorted(current_match["context_before"]):
                        code_lines.append(text)

                    code_lines.append(current_match["matched_text"])

                    for line_num, text in sorted(current_match["context_after"]):
                        code_lines.append(text)

                    code_snippet = "\n".join(code_lines)

                    # Add to results
                    results.append(
                        {
                            "file_path": current_match["file_path"],
                            "start_line": start_line,
                            "end_line": end_line,
                            "symbol_name": None,  # We don't have symbol information from ripgrep
                            "code": code_snippet,
                            "match_line": match_line,
                            "matched_text": current_match["matched_text"],
                            "similarity": 1.0,  # Exact match has perfect similarity
                            "source": "lexical",
                        }
                    )

                    current_match = None

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing ripgrep JSON output: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing ripgrep output: {e}")
                continue

        return results

    def search_function(self, function_name: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for function definitions.

        Args:
            function_name: Name of the function to search for
            **kwargs: Additional search parameters

        Returns:
            List of dictionaries with code snippets and metadata
        """
        # We can create language-specific patterns for common languages
        # but this is a simple approach that works across languages
        pattern = f"(function|def|class)\\s+{function_name}\\b"
        return self.search(pattern, **kwargs)

    def search_file(self, file_pattern: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for files matching a pattern.

        Args:
            file_pattern: Pattern to match against file names
            **kwargs: Additional search parameters

        Returns:
            List of dictionaries with file information
        """
        # Use find command to locate files
        cmd = ["find", str(self.repo_path), "-type", "f", "-name", file_pattern]

        try:
            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            file_paths = result.stdout.strip().split("\n")

            # Format results
            results = []
            for path in file_paths:
                if not path:
                    continue

                results.append(
                    {
                        "file_path": path,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol_name": None,
                        "code": f"File: {os.path.basename(path)}",
                        "similarity": 1.0,
                        "source": "file_search",
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Failed to search for files: {e}")
            return []


def create_lexical_search(config: CodeSleuthConfig) -> LexicalSearch:
    """
    Create a LexicalSearch instance from a configuration.

    Args:
        config: CodeSleuth configuration

    Returns:
        LexicalSearch instance
    """
    return LexicalSearch(config)
