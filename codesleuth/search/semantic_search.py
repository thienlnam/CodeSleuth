"""
Semantic search module for CodeSleuth.

This module provides semantic code search capabilities by leveraging
the vector embeddings in the semantic index.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

from loguru import logger

from ..config import CodeSleuthConfig
from ..indexing.parser import CodeChunk
from ..indexing.semantic_index import SemanticIndex, create_semantic_index


class SemanticSearch:
    """Semantic code search using vector embeddings."""

    def __init__(self, semantic_index: SemanticIndex):
        """
        Initialize the semantic search.

        Args:
            semantic_index: Semantic index for code search
        """
        self.semantic_index = semantic_index

    def is_available(self) -> bool:
        """
        Check if semantic search is available.

        Returns:
            True if semantic search is available, False otherwise
        """
        return self.semantic_index.is_semantic_search_available()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code snippets semantically similar to the query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of dictionaries with code snippets and metadata.
            Returns empty list if semantic search is not available.
        """
        logger.debug(f"Semantic search for: {query} (top_k={top_k})")

        # Check if semantic search is available
        if not self.is_available():
            logger.warning(
                "Semantic search was requested but is not available - returning empty results"
            )
            return []

        # Get search results from the index
        results = self.semantic_index.search(query, top_k=top_k)

        # Format results for consumption by LLMs or other clients
        formatted_results = []
        for chunk, similarity in results:
            formatted_results.append(
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol_name": chunk.symbol_name,
                    "code": chunk.code,
                    "similarity": float(similarity),  # Ensure it's serializable
                    "source": "semantic",
                }
            )

        logger.debug(f"Found {len(formatted_results)} semantic results")
        return formatted_results


def create_semantic_search(config: CodeSleuthConfig) -> SemanticSearch:
    """
    Create a SemanticSearch instance from a configuration.

    Args:
        config: CodeSleuth configuration

    Returns:
        SemanticSearch instance
    """
    semantic_index = create_semantic_index(config)
    return SemanticSearch(semantic_index)
