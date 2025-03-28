"""
API module for CodeSleuth.

This module provides a simple API for using CodeSleuth's search capabilities.
It combines semantic and lexical search for comprehensive code search.
"""

from typing import Dict, List, Optional, Union, Any
import os
import json
import subprocess

from loguru import logger

from .config import CodeSleuthConfig
from .indexing.parser import CodeParser
from .indexing.semantic_index import index_repository
from .search import (
    SemanticSearch,
    LexicalSearch,
    create_semantic_search,
    create_lexical_search,
)


class CodeSleuth:
    """Main API for CodeSleuth."""

    def __init__(self, config: CodeSleuthConfig):
        """
        Initialize CodeSleuth.

        Args:
            config: CodeSleuth configuration
        """
        self.config = config
        self.semantic_search = create_semantic_search(config)
        self.lexical_search = create_lexical_search(config)

    def is_semantic_search_available(self) -> bool:
        """
        Check if semantic search is available.

        Returns:
            True if semantic search is available, False otherwise
        """
        return self.semantic_search.is_available()

    def index_repository(self):
        """Index the repository."""
        logger.info(f"Indexing repository: {self.config.repo_path}")
        parser = CodeParser(self.config)
        index_repository(self.config, parser, self.semantic_search.semantic_index)
        logger.info("Repository indexed successfully")

    def search_semantically(
        self, query: str, top_k: int = 10, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for code semantically.

        Args:
            query: Query string
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0) for results.
                               Results below this threshold will be filtered out.

        Returns:
            List of search results. If semantic search is not available,
            falls back to lexical search with a warning.
        """
        if not self.is_semantic_search_available():
            logger.warning(
                "Semantic search unavailable, falling back to lexical search"
            )
            return self.search_lexically(query, max_results=top_k)

        results = self.semantic_search.search(query, top_k=top_k)

        # Filter results based on similarity threshold
        if similarity_threshold > 0.0:
            results = [
                result
                for result in results
                if result.get("similarity", 0.0) >= similarity_threshold
            ]

        return results

    def search_lexically(
        self,
        query: str,
        max_results: int = 50,
        case_sensitive: bool = False,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for code lexically using patterns.

        Args:
            query: Query string
            max_results: Maximum number of results to return
            case_sensitive: Whether the search should be case sensitive
            include_pattern: Glob pattern for files to include
            exclude_pattern: Glob pattern for files to exclude

        Returns:
            List of search results
        """
        return self.lexical_search.search(
            query,
            max_results=max_results,
            case_sensitive=case_sensitive,
            include_pattern=include_pattern,
            exclude_pattern=exclude_pattern,
        )

    def search_function_definitions(
        self,
        function_name: str,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for function definitions.

        Args:
            function_name: Name of the function to search for
            max_results: Maximum number of results to return

        Returns:
            List of search results containing function definitions
        """
        return self.lexical_search.search_function(
            function_name, max_results=max_results
        )

    def search_file(self, file_pattern: str) -> List[Dict[str, Any]]:
        """
        Search for files matching a pattern.

        Args:
            file_pattern: Pattern to match against file names

        Returns:
            List of file information
        """
        return self.lexical_search.search_file(file_pattern)

    def view_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        context_lines: int = 5,
    ) -> str:
        """
        Retrieve a snippet of file contents with optional line-range context.

        Args:
            file_path: Path to the file
            start_line: Starting line number (optional)
            end_line: Ending line number (optional)
            context_lines: Number of surrounding lines for context

        Returns:
            Snippet of file contents
        """
        # Check if path is absolute before any path operations
        if not os.path.isabs(file_path):
            file_path = os.path.join(str(self.config.repo_path), file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            # If no line range specified, return first few lines
            if start_line is None and end_line is None:
                return "".join(lines[: min(10, len(lines))])

            # Calculate valid line range
            total_lines = len(lines)

            if start_line is None:
                start_line = max(1, end_line - context_lines)
            if end_line is None:
                end_line = min(total_lines, start_line + context_lines)

            # Adjust line numbers to be 0-indexed for list access
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, end_line)

            # Add context if requested
            if context_lines > 0:
                start_idx = max(0, start_idx - context_lines)
                end_idx = min(total_lines, end_idx + context_lines)

            # Extract the requested lines and join them
            return "".join(lines[start_idx:end_idx])

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return f"Error reading file: {str(e)}"

    def get_project_structure(self) -> Dict[str, Any]:
        """
        Return a structured representation of the project files and directories.

        Returns:
            Nested dictionary representing the project file structure.
            On success, returns {"root": {<directory_structure>}}.
            On failure, returns {"error": <error_message>}.
        """

        def build_tree(path: str) -> Optional[Dict[str, Any]]:
            """Build a tree structure for the given path."""
            result = {}
            try:
                # Resolve any symlinks and get absolute path
                real_path = os.path.realpath(path)

                # Verify the path exists and is accessible
                if not os.path.exists(real_path):
                    logger.error(f"Path does not exist: {real_path}")
                    return None

                # Verify it's a directory
                if not os.path.isdir(real_path):
                    logger.error(f"Path is not a directory: {real_path}")
                    return None

                # Try to list directory contents
                try:
                    items = os.listdir(real_path)
                except PermissionError as e:
                    logger.error(
                        f"Permission denied accessing directory {real_path}: {e}"
                    )
                    return None
                except OSError as e:
                    logger.error(f"OS error accessing directory {real_path}: {e}")
                    return None

                # Process each item in the directory
                for item in sorted(items):  # Sort for consistent ordering
                    # Skip hidden files and directories
                    if item.startswith("."):
                        continue

                    full_path = os.path.join(real_path, item)

                    try:
                        # Handle symlinks by resolving them
                        if os.path.islink(full_path):
                            real_item_path = os.path.realpath(full_path)
                            # Skip if symlink points outside the repo
                            if not real_item_path.startswith(
                                str(self.config.repo_path)
                            ):
                                logger.debug(
                                    f"Skipping symlink that points outside repo: {full_path}"
                                )
                                continue
                            full_path = real_item_path

                        if os.path.isdir(full_path):
                            subtree = build_tree(full_path)
                            if subtree is not None:  # Only add non-empty directories
                                result[item] = subtree
                        else:
                            # Only include regular files
                            if os.path.isfile(full_path):
                                result[item] = None  # Files are leaf nodes
                    except (PermissionError, OSError) as e:
                        logger.error(f"Error accessing {full_path}: {e}")
                        continue

                return result if result else None  # Return None for empty directories

            except Exception as e:
                logger.error(f"Unexpected error building tree for {path}: {e}")
                return None

        try:
            repo_path = str(self.config.repo_path)

            # Initial repository checks
            if not os.path.exists(repo_path):
                msg = f"Repository path does not exist: {repo_path}"
                logger.error(msg)
                return {"error": msg}

            if not os.path.isdir(repo_path):
                msg = f"Repository path is not a directory: {repo_path}"
                logger.error(msg)
                return {"error": msg}

            # Try to access the repository
            try:
                os.listdir(repo_path)
            except PermissionError as e:
                msg = f"Permission denied accessing repository {repo_path}: {e}"
                logger.error(msg)
                return {"error": msg}
            except OSError as e:
                msg = f"OS error accessing repository {repo_path}: {e}"
                logger.error(msg)
                return {"error": msg}

            # Build the tree structure
            tree = build_tree(repo_path)
            if not tree:
                msg = f"No files found in repository: {repo_path}"
                logger.error(msg)
                return {"error": msg}

            return {"root": tree}

        except Exception as e:
            msg = f"Failed to get project structure: {e}"
            logger.error(msg)
            return {"error": msg}

    def get_code_metadata(self, file_path: str) -> Dict[str, List[str]]:
        """
        Get metadata like functions and classes defined in a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with keys ('functions', 'classes') listing names
        """
        # Resolve absolute path if necessary
        if not os.path.isabs(file_path):
            file_path = os.path.join(str(self.config.repo_path), file_path)

        logger.debug(f"Searching for metadata in file: {file_path}")
        logger.debug(f"File exists: {os.path.exists(file_path)}")

        # Search for functions using ripgrep with generic patterns
        function_patterns = [
            # Generic function/method definitions with context
            r"(?m)^[ \t]*(?:def|function|fn|func)\s+\w+[^{;]*(?:\{|:)(?:\s*\n\s*[^def\n][^\n]*)*",  # Common function keywords
            r"(?m)^[ \t]*(?:public|private|protected|static|async|virtual|inline)\s+(?:const\s+)?(?:\w+[&*\s]+)*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:\{|:)(?:\s*\n\s*[^(public|private|protected|static|async|virtual|inline)\n][^\n]*)*",  # Method modifiers (including C++)
            r"(?m)^[ \t]*(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>.*(?:\{|:)(?:\s*\n\s*[^(const|let|var)\n][^\n]*)*",  # Arrow functions
            r"(?m)^[ \t]*(?:\w+[&*\s]+)*\w+\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:\{|:)(?:\s*\n\s*[^\n]*)*",  # C++ free functions
            r"(?m)^[ \t]*(?:function)\s+\w+\s*\([^)]*\)\s*(?:\{|:)(?:\s*\n\s*[^function\n][^\n]*)*",  # PHP functions
        ]

        # Search for classes using generic patterns
        class_patterns = [
            # C++ template class pattern
            r"(?m)^[ \t]*(?:template\s*<\s*typename\s+\w+\s*>\s*)?(?:class)\s+\w+(?:\s*:\s*(?:public|private|protected)\s+\w+(?:<[^>]+>)?(?:\s*,\s*(?:public|private|protected)\s+\w+(?:<[^>]+>)?)*)?[^{]*\{(?:[^{}]*|\{(?:[^{}]*|\{[^{}]*\})*\})*\}",
            # Other patterns
            r"(?m)^[ \t]*(?:class)\s+\w+[^{;]*(?:\{|:)(?:\s*\n\s*[^(class)\n][^\n]*)*",
            r"(?m)^[ \t]*(?:export\s+)?(?:abstract\s+)?(?:class|interface)\s+\w+(?:\s*:\s*(?:public|private|protected)\s+\w+)?(?:\s+implements\s+\w+(?:\s*,\s*\w+)*)?[^{;]*(?:\{|:)(?:\s*\n\s*[^(class|interface)\n][^\n]*)*",
            r"(?m)^[ \t]*(?:pub\s+)?(?:struct|class)\s+\w+(?:\s*:\s*(?:public|private|protected)\s+\w+)?[^{;]*(?:\{|:)(?:\s*\n\s*[^(struct|class)\n][^\n]*)*",
            r"(?m)^[ \t]*(?:class|interface|trait)\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+(?:\s*,\s*\w+)*)?[^{;]*(?:\{|:)(?:\s*\n\s*[^(class|interface|trait)\n][^\n]*)*",
        ]

        metadata = {"functions": [], "classes": []}

        try:
            # Search for functions with smart case and multiline support
            for pattern in function_patterns:
                function_cmd = [
                    "rg",
                    "--smart-case",  # Smart case matching
                    "--json",  # JSON output for reliable parsing
                    "--multiline",  # Handle multiline matches
                    "--only-matching",  # Only return the matched pattern
                    "--multiline-dotall",  # Make . match newlines
                    pattern,
                    file_path,
                ]
                logger.debug(f"Running ripgrep command: {' '.join(function_cmd)}")
                function_result = subprocess.run(
                    function_cmd, text=True, capture_output=True, check=False
                )
                logger.debug(f"Ripgrep return code: {function_result.returncode}")
                logger.debug(f"Ripgrep stdout: {function_result.stdout}")
                logger.debug(f"Ripgrep stderr: {function_result.stderr}")

                # Parse function results
                if function_result.returncode in [
                    0,
                    1,
                ]:  # 0=matches found, 1=no matches
                    for line in function_result.stdout.splitlines():
                        if not line or not line.strip():
                            continue

                        try:
                            data = json.loads(line.strip())
                            if data.get("type") == "match":
                                match_data = data.get("data", {})
                                line_text = (
                                    match_data.get("lines", {}).get("text", "").strip()
                                )
                                logger.debug(f"Found function: {line_text}")

                                # Skip empty or private functions
                                if not line_text or line_text.lstrip().startswith("_"):
                                    continue

                                # Add the signature if not already present
                                if line_text not in metadata["functions"]:
                                    metadata["functions"].append(line_text)
                        except json.JSONDecodeError:
                            continue

            # Search for classes
            for pattern in class_patterns:
                class_cmd = [
                    "rg",
                    "--smart-case",  # Smart case matching
                    "--json",  # JSON output for reliable parsing
                    "--multiline",  # Handle multiline matches
                    "--only-matching",  # Only return the matched pattern
                    "--multiline-dotall",  # Make . match newlines
                    pattern,
                    file_path,
                ]
                logger.debug(f"Running ripgrep command: {' '.join(class_cmd)}")
                class_result = subprocess.run(
                    class_cmd, text=True, capture_output=True, check=False
                )
                logger.debug(f"Ripgrep return code: {class_result.returncode}")
                logger.debug(f"Ripgrep stdout: {class_result.stdout}")
                logger.debug(f"Ripgrep stderr: {class_result.stderr}")

                # Parse class results
                if class_result.returncode in [0, 1]:  # 0=matches found, 1=no matches
                    for line in class_result.stdout.splitlines():
                        if not line or not line.strip():
                            continue

                        try:
                            data = json.loads(line.strip())
                            if data.get("type") == "match":
                                match_data = data.get("data", {})
                                line_text = (
                                    match_data.get("lines", {}).get("text", "").strip()
                                )
                                logger.debug(f"Found class: {line_text}")

                                # Skip empty or private classes
                                if not line_text or line_text.lstrip().startswith("_"):
                                    continue

                                # Add the signature if not already present
                                if line_text not in metadata["classes"]:
                                    metadata["classes"].append(line_text)
                        except json.JSONDecodeError:
                            continue

            logger.debug(f"Final metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to get code metadata for {file_path}: {e}")
            return metadata

    def search_references(
        self, symbol: str, definition: bool = False, max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search where a given symbol (function, variable, class) is referenced or defined.

        Args:
            symbol: Symbol name to search for
            definition: If True, search definitions only; otherwise, find references
            max_results: Maximum number of results to return

        Returns:
            List of references or definitions with file and line information
        """
        if definition:
            # Search for definitions (more specific patterns)
            patterns = [
                f"(function|def)\\s+{symbol}\\s*\\(",  # Function definition
                f"class\\s+{symbol}\\b",  # Class definition
                f"\\b(const|let|var)\\s+{symbol}\\b",  # Variable definition (JS/TS)
                f"\\b[a-zA-Z_][a-zA-Z0-9_]*\\s+{symbol}\\b",  # Variable definition (other langs)
            ]

            all_results = []
            for pattern in patterns:
                results = self.search_lexically(
                    pattern, max_results=max_results, case_sensitive=True
                )
                all_results.extend(results)

            # Deduplicate and limit results
            seen = set()
            filtered_results = []
            for result in all_results:
                key = (result["file_path"], result["start_line"])
                if key not in seen:
                    seen.add(key)
                    filtered_results.append(result)
                    if len(filtered_results) >= max_results:
                        break

            return filtered_results
        else:
            # Search for all references of the symbol
            return self.search_lexically(
                f"\\b{symbol}\\b", max_results=max_results, case_sensitive=True
            )


def create_codesleuth(config: CodeSleuthConfig) -> CodeSleuth:
    """
    Create a CodeSleuth instance from a configuration.

    Args:
        config: CodeSleuth configuration

    Returns:
        CodeSleuth instance
    """
    return CodeSleuth(config)
