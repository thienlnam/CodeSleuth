"""
API module for CodeSleuth.

This module provides a simple API for using CodeSleuth's search capabilities.
It combines semantic and lexical search for comprehensive code search.
"""

from typing import Dict, List, Optional, Union, Any

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

    def search_semantically(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code semantically.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of search results. If semantic search is not available,
            falls back to lexical search with a warning.
        """
        if not self.is_semantic_search_available():
            logger.warning(
                "Semantic search unavailable, falling back to lexical search"
            )
            return self.search_lexically(query, max_results=top_k)

        return self.semantic_search.search(query, top_k=top_k)

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

    def search_function(
        self, function_name: str, max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for function definitions.

        Args:
            function_name: Name of the function to search for
            max_results: Maximum number of results to return

        Returns:
            List of search results
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

    def hybrid_search(
        self, query: str, top_k: int = 10, max_lexical: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search combining semantic and lexical results.

        Args:
            query: Query string
            top_k: Number of final results to return
            max_lexical: Maximum number of lexical results to include

        Returns:
            List of search results
        """
        # Get semantic results if available
        if self.is_semantic_search_available():
            semantic_results = self.search_semantically(query, top_k=top_k)
        else:
            logger.warning(
                "Semantic search unavailable for hybrid search, using only lexical results"
            )
            semantic_results = []

        # Get lexical results for the same query
        lexical_results = self.search_lexically(query, max_results=max_lexical)

        # If no semantic results, just return lexical results up to top_k
        if not semantic_results:
            return lexical_results[:top_k]

        # Combine results, prioritizing semantic results but ensuring some lexical ones
        # This is a simple approach - could be made more sophisticated
        combined = semantic_results.copy()

        # Track seen file paths and line ranges to avoid duplicates
        seen = set()
        for result in combined:
            key = (
                result["file_path"],
                result["start_line"],
                result["end_line"],
            )
            seen.add(key)

        # Add lexical results that don't overlap with semantic results
        for result in lexical_results:
            key = (
                result["file_path"],
                result["start_line"],
                result["end_line"],
            )
            if key not in seen:
                combined.append(result)
                seen.add(key)
                # Stop if we've reached the desired number of results
                if len(combined) >= top_k:
                    break

        # Sort by similarity score
        combined.sort(key=lambda x: x["similarity"], reverse=True)

        # Limit to top_k
        return combined[:top_k]


def create_codesleuth(config: CodeSleuthConfig) -> CodeSleuth:
    """
    Create a CodeSleuth instance from a configuration.

    Args:
        config: CodeSleuth configuration

    Returns:
        CodeSleuth instance
    """
    return CodeSleuth(config)
