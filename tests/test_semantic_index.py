import os
import pytest
import numpy as np
from pathlib import Path

from codesleuth.config import CodeSleuthConfig, IndexConfig, EmbeddingModel
from codesleuth.indexing.parser import CodeChunk
from codesleuth.indexing.semantic_index import (
    SemanticIndex,
    CodeEmbedder,
    create_semantic_index,
    MODEL_REGISTRY,
)


@pytest.fixture(scope="session")
def mock_embedder():
    """Create a mock embedder that returns fixed embeddings."""

    class MockEmbedder(CodeEmbedder):
        def __init__(self, *args, **kwargs):
            # Skip actual model loading
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
            self.use_mlx = False
            # Get the model info to determine embedding dimension
            if args and isinstance(args[0], EmbeddingModel):
                self.model_info = MODEL_REGISTRY[args[0]]
            else:
                # Default to CodeBERT dimension if no model specified
                self.model_info = MODEL_REGISTRY[EmbeddingModel.CODEBERT]

        def embed(self, text):
            """Return a deterministic embedding based on text length."""
            # Create a deterministic embedding vector based on text length
            # This allows us to test similarity search without a real model
            vector_size = self.model_info.embedding_dim
            base_vector = np.ones(vector_size) * len(text) % 100

            # Add some noise to make vectors unique but deterministically based on text
            # Make sure the noise vector is the correct length
            text_bytes = text.encode("utf-8")[:100]  # Take up to 100 bytes
            noise = np.zeros(vector_size)

            # Fill in what we can from the text bytes
            for i, byte_val in enumerate(text_bytes):
                if i < vector_size:
                    noise[i] = byte_val % 10

            return (base_vector + noise).astype(np.float32)

        def embed_batch(self, texts, batch_size=32):
            """Embed a batch of texts."""
            return np.array([self.embed(text) for text in texts])

    return MockEmbedder()


@pytest.fixture(scope="session")
def test_index_config(tmp_path_factory):
    """Create a test configuration for the semantic index."""
    temp_dir = tmp_path_factory.mktemp("index")
    config = IndexConfig()
    config.index_path = temp_dir
    config.model_name = EmbeddingModel.CODEBERT
    config.use_mlx = False  # Disable MLX for testing
    # Use smaller HNSW values for faster testing
    config.hnsw_m = 8
    config.hnsw_ef_construction = 64
    config.hnsw_ef_search = 32
    return config


@pytest.fixture(scope="session")
def semantic_index(test_index_config, mock_embedder):
    """Create a semantic index with mock embedder for testing."""
    index = SemanticIndex(test_index_config)
    # Replace the real embedder with our mock
    index.embedder = mock_embedder
    return index


@pytest.fixture(scope="session")
def sample_code_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            file_path="test/example1.py",
            start_line=1,
            end_line=10,
            code="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            symbol_name="fibonacci",
            language="python",
        ),
        CodeChunk(
            file_path="test/example2.py",
            start_line=1,
            end_line=8,
            code="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            symbol_name="factorial",
            language="python",
        ),
    ]


def test_add_chunks(semantic_index, sample_code_chunks):
    """Test adding chunks to the index."""
    # Start with empty index
    initial_count = semantic_index.index.ntotal if semantic_index.index else 0

    # Add chunks
    semantic_index.add_chunks(sample_code_chunks)

    # Verify chunks were added
    assert semantic_index.index.ntotal == initial_count + len(sample_code_chunks)

    # Verify metadata was stored
    for chunk in sample_code_chunks:
        # Find the chunk in metadata
        found = False
        for id, entry in semantic_index.metadata.items():
            if (
                entry.chunk.file_path == chunk.file_path
                and entry.chunk.symbol_name == chunk.symbol_name
            ):
                found = True
                break
        assert (
            found
        ), f"Chunk {chunk.file_path}:{chunk.symbol_name} not found in metadata"


def test_search(semantic_index, sample_code_chunks):
    """Test searching the index."""
    # Make sure chunks are in the index
    if semantic_index.index.ntotal < len(sample_code_chunks):
        semantic_index.add_chunks(sample_code_chunks)

    # Search for specific terms
    fibonacci_results = semantic_index.search("fibonacci recursive function", 3)
    assert len(fibonacci_results) > 0

    # At least one result should contain the fibonacci function
    fibonacci_found = any(
        "fibonacci" in chunk.code.lower() for chunk, score in fibonacci_results
    )
    assert fibonacci_found

    # Check scores are in descending order (higher similarity first)
    scores = [score for _, score in fibonacci_results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_remove_file(semantic_index, sample_code_chunks):
    """Test removing chunks for a file."""
    # Make sure chunks are in the index
    if semantic_index.index.ntotal < len(sample_code_chunks):
        semantic_index.add_chunks(sample_code_chunks)

    # Count chunks for a specific file
    file_path = "test/example1.py"
    initial_count = sum(
        1
        for id, entry in semantic_index.metadata.items()
        if entry.chunk.file_path == file_path
    )

    # Remove chunks for the file
    semantic_index.remove_file(file_path)

    # Verify chunks were removed
    final_count = sum(
        1
        for id, entry in semantic_index.metadata.items()
        if entry.chunk.file_path == file_path
    )
    assert final_count == 0, f"Expected 0 chunks for {file_path}, found {final_count}"


def test_update_file(semantic_index, sample_code_chunks):
    """Test updating chunks for a file."""
    # Make sure chunks are in the index
    if semantic_index.index.ntotal < len(sample_code_chunks):
        semantic_index.add_chunks(sample_code_chunks)

    # Create updated chunks for a file
    file_path = "test/example2.py"
    updated_chunks = [
        CodeChunk(
            file_path=file_path,
            start_line=1,
            end_line=8,
            code="def improved_factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
            symbol_name="improved_factorial",
            language="python",
        )
    ]

    # Update file
    semantic_index.update_file(file_path, updated_chunks)

    # Verify only the new chunks are present
    chunks = [
        entry.chunk
        for id, entry in semantic_index.metadata.items()
        if entry.chunk.file_path == file_path
    ]
    assert len(chunks) == len(
        updated_chunks
    ), f"Expected {len(updated_chunks)} chunks, found {len(chunks)}"
    assert (
        chunks[0].symbol_name == "improved_factorial"
    ), "Expected updated chunk content"


def test_create_semantic_index():
    """Test the create_semantic_index factory function."""
    config = CodeSleuthConfig()
    index = create_semantic_index(config)
    assert isinstance(index, SemanticIndex)
    assert index.config == config.index
