"""
Configuration module for CodeSleuth.

This module defines the configuration classes for the CodeSleuth code search tool.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field


class EmbeddingModel(str, Enum):
    """Enum for supported embedding models."""

    CODEBERT = "codebert"
    E5_SMALL = "e5-small-v2"
    BGE_M3 = "bge-m3"


class VectorStore(str, Enum):
    """Enum for supported vector stores."""

    FAISS = "faiss"


class ParserConfig(BaseModel):
    """Configuration for the code parser."""

    chunk_size: int = Field(
        default=100,
        description="Maximum number of lines in a code chunk",
    )
    chunk_overlap: int = Field(
        default=20,
        description="Number of lines to overlap between chunks",
    )
    ignore_patterns: List[str] = Field(
        default=["node_modules", "dist", "build", ".git", "__pycache__", "*.pyc"],
        description="Patterns to ignore when scanning directories",
    )
    respect_gitignore: bool = Field(
        default=True,
        description="Whether to respect .gitignore files",
    )
    extension_to_language: Dict[str, str] = Field(
        default={
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".php": "php",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".txt": "text",
            ".md": "text",
            ".json": "text",
            ".yaml": "text",
            ".yml": "text",
            ".toml": "text",
            ".ini": "text",
            ".cfg": "text",
            ".conf": "text",
        },
        description="Mapping of file extensions to language names",
    )


class IndexConfig(BaseModel):
    """Configuration for the semantic index."""

    model_name: EmbeddingModel = Field(
        default=EmbeddingModel.BGE_M3,
        description="Model to use for embeddings",
    )
    vector_store: VectorStore = Field(
        default=VectorStore.FAISS,
        description="Vector store to use",
    )
    index_path: Path = Field(
        default=Path("./codesleuth_index"),
        description="Path to store the index",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for computing embeddings",
    )
    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU for embedding computation",
    )
    use_mlx: bool = Field(
        default=True,
        description="Whether to use MLX for embedding computation on Apple Silicon. "
        "If True, will automatically detect and use MLX when available on Apple Silicon. "
        "Set to False to force PyTorch even on Apple Silicon.",
    )
    # HNSW index configuration
    hnsw_m: int = Field(
        default=16,
        description="Number of connections per node in HNSW index (higher = better recall, slower construction)",
    )
    hnsw_ef_construction: int = Field(
        default=100,
        description="Search depth during construction in HNSW index (higher = better recall, slower construction)",
    )
    hnsw_ef_search: int = Field(
        default=64,
        description="Search depth during search in HNSW index (higher = better recall, slower search)",
    )


class SearchConfig(BaseModel):
    """Configuration for code search."""

    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
    )
    min_similarity: float = Field(
        default=0.5,
        description="Minimum similarity score for results (0-1)",
    )
    max_grep_results: int = Field(
        default=50,
        description="Maximum number of results to return from grep search",
    )


class CodeSleuthConfig(BaseModel):
    """Main configuration for CodeSleuth."""

    repo_path: Path = Field(
        default=Path.cwd(),
        description="Path to the repository to index",
    )
    parser: ParserConfig = Field(
        default_factory=ParserConfig,
        description="Parser configuration",
    )
    index: IndexConfig = Field(
        default_factory=IndexConfig,
        description="Index configuration",
    )
    search: SearchConfig = Field(
        default_factory=SearchConfig,
        description="Search configuration",
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


def load_config(config_path: Optional[Union[str, Path]] = None) -> CodeSleuthConfig:
    """
    Load configuration from a file or create a default configuration.

    Args:
        config_path: Path to configuration file (YAML/JSON)

    Returns:
        CodeSleuthConfig: Configuration object
    """
    # TODO: Implement loading from YAML/JSON
    return CodeSleuthConfig()
