"""
Semantic index module for CodeSleuth.

This module handles the embedding of code chunks and indexing them in a vector database.
It supports incremental updates to the index when code files change.
"""

import os
import pickle
import sys
import platform
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import faiss
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..config import CodeSleuthConfig, EmbeddingModel, IndexConfig
from .parser import CodeChunk, CodeParser


@dataclass
class ModelInfo:
    """Information about an embedding model."""

    name: str
    huggingface_name: str
    embedding_dim: int
    mlx_name: Optional[str] = None  # None means not supported on MLX
    max_length: int = 512


# Simple registry of supported models
MODEL_REGISTRY = {
    EmbeddingModel.CODEBERT: ModelInfo(
        name="codebert",
        huggingface_name="microsoft/codebert-base",
        mlx_name=None,  # CodeBERT not supported on MLX
        embedding_dim=768,
    ),
    EmbeddingModel.E5_SMALL: ModelInfo(
        name="e5-small-v2",
        huggingface_name="intfloat/e5-small-v2",
        mlx_name=None,  # E5-small not supported on MLX
        embedding_dim=384,
    ),
    EmbeddingModel.BGE_M3: ModelInfo(
        name="bge-m3",
        huggingface_name="BAAI/bge-m3",
        mlx_name="bge-m3",  # BGE-M3 is supported on MLX
        embedding_dim=1024,  # BGE-M3 has larger embeddings
        max_length=512,
    ),
}


# Import MLX Embedding Models
try:
    from mlx_embedding_models.embedding import EmbeddingModel as MLXEmbeddingModel

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX Embedding Models not available, falling back to PyTorch")


@dataclass
class IndexEntry:
    """Entry in the semantic index."""

    id: int
    chunk: CodeChunk
    embedding: Optional[np.ndarray] = None


class CodeEmbedder:
    """Code embedding model."""

    def __init__(
        self, model_name: EmbeddingModel, use_gpu: bool = False, use_mlx: bool = True
    ):
        """
        Initialize the code embedder.

        Args:
            model_name: Name of the model to use
            use_gpu: Whether to use GPU for embedding computation
            use_mlx: Whether to use MLX for embedding computation on Apple Silicon
                    If True, will automatically detect and use MLX on Apple Silicon if available

        Raises:
            ValueError: If the model is not supported or not available on the current platform
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_info = MODEL_REGISTRY[model_name]
        self.use_mlx = False  # Default to not using MLX

        # Check if we should use MLX
        is_apple_silicon = platform.processor() == "arm"
        if is_apple_silicon and MLX_AVAILABLE and use_mlx and not use_gpu:
            if self.model_info.mlx_name is not None:
                logger.info(f"Using MLX model: {self.model_info.mlx_name}")
                self.use_mlx = True
                self.device = None  # MLX handles device management
                self.model = MLXEmbeddingModel.from_registry(self.model_info.mlx_name)
                self.tokenizer = None  # MLX handles tokenization
                return
            else:
                logger.info("MLX not available for this model, falling back to PyTorch")

        # If we're not using MLX, use PyTorch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        logger.info(
            f"Using HuggingFace model: {self.model_info.huggingface_name} on {self.device}"
        )

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.huggingface_name)
        self.model = AutoModel.from_pretrained(self.model_info.huggingface_name).to(
            self.device
        )
        self.model.eval()

    def embed(self, code: str) -> np.ndarray:
        """
        Embed a code snippet.

        Args:
            code: Code snippet to embed

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding fails
        """
        try:
            if self.use_mlx:
                # MLX handles tokenization and embedding
                embeddings = self.model.encode([code])
                return embeddings[0]
            else:
                # PyTorch path
                inputs = self.tokenizer(
                    code,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_info.max_length,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                return embeddings[0]
        except Exception as e:
            logger.error(f"Error embedding code: {e}")
            raise

    def embed_batch(self, codes: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of code snippets.

        Args:
            codes: List of code snippets to embed
            batch_size: Batch size for embedding computation

        Returns:
            Array of embedding vectors

        Raises:
            RuntimeError: If batch embedding fails
        """
        if not codes:
            return np.array([])

        if self.use_mlx:
            # Process in batches for MLX to avoid memory issues
            all_embeddings = []
            for i in range(0, len(codes), batch_size):
                batch = codes[i : min(i + batch_size, len(codes))]
                try:
                    # MLX handles tokenization and embedding
                    batch_embeddings = self.model.encode(batch)

                    # Convert to numpy array if needed
                    if not isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = np.array(batch_embeddings)

                    # Ensure correct shape
                    if len(batch_embeddings.shape) == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                    elif len(batch_embeddings.shape) > 2:
                        # If we have more than 2 dimensions, take the first embedding
                        batch_embeddings = batch_embeddings[:, 0, :]

                    # Ensure float32 type and correct shape
                    batch_embeddings = batch_embeddings.astype(np.float32)

                    # Verify embedding dimension
                    if batch_embeddings.shape[1] != self.model_info.embedding_dim:
                        logger.error(
                            f"Unexpected embedding dimension: {batch_embeddings.shape[1]}, expected {self.model_info.embedding_dim}"
                        )
                        continue

                    all_embeddings.append(batch_embeddings)
                    logger.debug(
                        f"Successfully embedded batch {i//batch_size + 1}, shape: {batch_embeddings.shape}"
                    )
                except Exception as e:
                    logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                    continue

            # Stack all batches
            if all_embeddings:
                try:
                    stacked = np.vstack(all_embeddings)
                    logger.debug(
                        f"Successfully stacked all embeddings, final shape: {stacked.shape}"
                    )
                    return stacked
                except Exception as e:
                    logger.error(f"Error stacking embeddings: {e}")
                    raise
            return np.array([])
        else:
            all_embeddings = []

            for i in range(0, len(codes), batch_size):
                batch = codes[i : min(i + batch_size, len(codes))]

                # Tokenize the batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_info.max_length,
                ).to(self.device)

                # Compute embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(embeddings)

            return np.vstack(all_embeddings) if all_embeddings else np.array([])


class SemanticIndex:
    """Semantic index for code search."""

    def __init__(self, config: IndexConfig):
        """
        Initialize the semantic index.

        Args:
            config: Index configuration
        """
        self.config = config
        self.index_path = config.index_path
        self.embedding_available = True

        # Initialize the embedder
        try:
            self.embedder = CodeEmbedder(
                model_name=config.model_name,
                use_gpu=config.use_gpu,  # Pass use_gpu from config
                use_mlx=config.use_mlx,  # Pass use_mlx from config
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            logger.warning(
                "Semantic search will not be available - only lexical search will work"
            )
            self.embedding_available = False
            self.embedder = None

        # Initialize FAISS index
        self.index = None
        self.metadata: Dict[int, CodeChunk] = {}
        self.next_id = 0
        self.reusable_ids = set()  # Track IDs that can be reused

        # Load index if it exists
        if self._index_exists():
            try:
                self._load_index()
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                self._create_index()
        else:
            self._create_index()

    def is_semantic_search_available(self) -> bool:
        """
        Check if semantic search is available.

        Returns:
            True if semantic search is available, False otherwise
        """
        return self.embedding_available and self.embedder is not None

    def _index_exists(self) -> bool:
        """
        Check if the index exists.

        Returns:
            True if the index exists, False otherwise
        """
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.pkl"
        return index_file.exists() and metadata_file.exists()

    def _create_index(self):
        """Create a new FAISS index."""
        # Create directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Get embedding dimension from model registry
        embedding_dim = MODEL_REGISTRY[self.config.model_name].embedding_dim

        # Detect M1/M2 Macs for compatibility issues
        is_apple_silicon = platform.processor() == "arm"

        logger.debug(f"Creating index with embedding_dim={embedding_dim}")
        logger.debug(
            f"Python version: {sys.version_info.major}.{sys.version_info.minor}"
        )
        logger.debug(
            f"Platform processor: {platform.processor()}, Apple Silicone: {is_apple_silicon}"
        )

        try:
            if is_apple_silicon:
                # Known issue with HNSW on Apple Silicon
                logger.info(
                    "Detected Apple Silicone, using FlatL2 index for compatibility"
                )
                flat_index = faiss.IndexFlatL2(embedding_dim)
                logger.debug(f"Created FlatL2 index: {flat_index}")
                self.index = faiss.IndexIDMap(flat_index)
                logger.debug(f"Wrapped with IDMap: {self.index}")
            else:
                # Try to create HNSW index for better performance
                logger.info(
                    f"Creating HNSW index with M={self.config.hnsw_m}, efConstruction={self.config.hnsw_ef_construction}"
                )

                # Create the HNSW index with the specified parameters
                hnsw_index = faiss.IndexHNSWFlat(
                    embedding_dim, self.config.hnsw_m  # Number of connections per layer
                )

                # Set the HNSW construction-time and search-time parameters
                hnsw_index.hnsw.efConstruction = self.config.hnsw_ef_construction
                hnsw_index.hnsw.efSearch = self.config.hnsw_ef_search

                # Wrap with IDMap to support add_with_ids
                self.index = faiss.IndexIDMap(hnsw_index)

            # Verify index is initialized correctly
            logger.debug(f"Index type: {type(self.index)}")
            logger.debug(f"Index dimension: {self.index.d}")
            logger.debug(f"Index is trained: {self.index.is_trained}")
            logger.debug(f"Index total vectors: {self.index.ntotal}")

        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}", exc_info=True)
            # Fallback to FlatL2 if creation fails
            flat_index = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIDMap(flat_index)
            logger.info("Created FlatL2 index as fallback")

        # Use a more robust search method for problematic environments
        self.use_direct_search = not is_apple_silicon

    def _save_index(self):
        """Save the index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Log embedding status
        has_embeddings = any(
            isinstance(entry, IndexEntry) and entry.embedding is not None
            for entry in self.metadata.values()
        )

        if has_embeddings:
            logger.info("Saving metadata with embeddings for manual search")

        # Save metadata and reusable IDs
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump((self.metadata, self.next_id, self.reusable_ids), f)

        logger.info(
            f"Saved index with {self.index.ntotal} vectors to {self.index_path}"
        )

    def _load_index(self):
        """Load the index and metadata from disk."""
        try:
            # Load FAISS index
            index_file = self.index_path / "index.faiss"
            self.index = faiss.read_index(str(index_file))

            # Check if inner index is HNSW and set parameters
            if hasattr(self.index, "index"):
                inner_index = self.index.index
                if hasattr(inner_index, "hnsw"):
                    # Ensure efSearch is set according to config
                    inner_index.hnsw.efSearch = self.config.hnsw_ef_search

            # Load metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, "rb") as f:
                loaded_data = pickle.load(f)

                # Handle different metadata formats
                if isinstance(loaded_data, tuple):
                    if len(loaded_data) == 3:
                        self.metadata, self.next_id, self.reusable_ids = loaded_data
                    elif len(loaded_data) == 2:
                        self.metadata, self.next_id = loaded_data
                        self.reusable_ids = set()
                else:
                    self.metadata = loaded_data
                    self.next_id = max(loaded_data.keys()) + 1 if loaded_data else 0
                    self.reusable_ids = set()

            logger.info(
                f"Loaded index with {self.index.ntotal} vectors from {self.index_path}"
            )

            # Check if we have IndexEntry objects with embeddings in metadata
            first_entry = next(iter(self.metadata.values())) if self.metadata else None
            has_embeddings = (
                isinstance(first_entry, IndexEntry)
                and first_entry.embedding is not None
            )

            # Set flag for using direct search based on entry type
            if hasattr(self.index, "index"):
                inner_index = self.index.index
                use_hnsw = isinstance(inner_index, faiss.IndexHNSWFlat)
                self.use_direct_search = use_hnsw and not has_embeddings
            else:
                self.use_direct_search = False

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def _get_next_id(self) -> int:
        """
        Get the next available ID, preferring reusable IDs if available.

        Returns:
            ID to use for the next vector
        """
        if self.reusable_ids:
            return self.reusable_ids.pop()
        return self.next_id

    def add_chunks(self, chunks: List[CodeChunk]):
        """
        Add chunks to the index.

        Args:
            chunks: List of code chunks to add
        """
        if not chunks:
            logger.info("No chunks to add")
            return

        if not self.is_semantic_search_available():
            logger.warning("Embedding model not available - skipping semantic indexing")
            return

        try:
            # Extract code snippets
            codes = [chunk.code for chunk in chunks]
            logger.debug(f"Processing {len(codes)} code snippets")

            # Compute embeddings in batches
            logger.info(f"Computing embeddings for {len(codes)} chunks")
            try:
                embeddings = self.embedder.embed_batch(
                    codes, batch_size=self.config.batch_size
                )
                if embeddings is None or len(embeddings) == 0:
                    logger.error("No embeddings were generated")
                    return
                logger.debug(
                    f"Successfully computed embeddings: shape={embeddings.shape}"
                )
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}", exc_info=True)
                return

            # Ensure embeddings are in the correct format for FAISS
            if self.embedder.use_mlx:
                # MLX embeddings are already in the correct format
                embeddings = np.array(embeddings, dtype=np.float32)
                logger.debug(f"MLX embeddings shape: {embeddings.shape}")
            else:
                # PyTorch embeddings need to be converted to float32
                embeddings = embeddings.astype(np.float32)
                logger.debug(f"PyTorch embeddings shape: {embeddings.shape}")

            # Verify we have valid embeddings
            if embeddings.shape[0] == 0:
                logger.error("No valid embeddings to add to index")
                return

            # Assign IDs, preferring reusable IDs when available
            ids = []
            id_to_chunk = {}
            id_to_embedding = {}

            for i, chunk in enumerate(chunks):
                if i >= len(embeddings):
                    logger.warning(f"Skipping chunk {i} - no embedding available")
                    continue

                chunk_id = self._get_next_id()
                ids.append(chunk_id)
                id_to_chunk[chunk_id] = chunk
                id_to_embedding[chunk_id] = embeddings[i]

                # Only increment next_id if we used it
                if chunk_id == self.next_id:
                    self.next_id += 1

            if not ids:
                logger.error("No valid chunks to add to index")
                return

            # Add to index
            logger.debug(f"Adding {len(ids)} vectors to index")
            try:
                # Debug info about embeddings
                logger.debug(f"Embeddings dtype: {embeddings.dtype}")
                logger.debug(f"Embeddings shape: {embeddings.shape}")
                logger.debug(f"Embeddings memory layout: {embeddings.flags}")
                logger.debug(f"Any NaN in embeddings: {np.isnan(embeddings).any()}")
                logger.debug(f"Any Inf in embeddings: {np.isinf(embeddings).any()}")

                # Debug info about IDs
                ids_array = np.array(ids)
                logger.debug(f"IDs dtype: {ids_array.dtype}")
                logger.debug(f"IDs shape: {ids_array.shape}")
                logger.debug(f"IDs: {ids_array}")

                # Ensure embeddings are contiguous and in the right format
                embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
                ids_array = np.ascontiguousarray(ids_array, dtype=np.int64)

                # Verify dimensions match
                if len(ids_array) != len(embeddings):
                    logger.error(
                        f"Dimension mismatch: ids={len(ids_array)}, embeddings={len(embeddings)}"
                    )
                    raise ValueError(
                        "Number of IDs does not match number of embeddings"
                    )

                self.index.add_with_ids(embeddings, ids_array)
                logger.debug(
                    f"Successfully added vectors to index, new total: {self.index.ntotal}"
                )
            except Exception as e:
                logger.error(f"Error adding vectors to index: {e}", exc_info=True)
                raise

            # Update metadata
            for chunk_id, chunk in id_to_chunk.items():
                # Create an IndexEntry that includes the embedding
                embedding = id_to_embedding.get(chunk_id)
                self.metadata[chunk_id] = IndexEntry(
                    id=chunk_id, chunk=chunk, embedding=embedding
                )

            # Save index
            self._save_index()

            logger.info(f"Added {len(chunks)} chunks to index")
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}", exc_info=True)
            logger.warning("Semantic search functionality may be limited")

    def remove_file(self, file_path: Union[str, Path]):
        """
        Remove all chunks for a file from the index.

        Args:
            file_path: Path to the file
        """
        str_path = str(file_path)

        # Find chunks to remove
        to_remove = []
        for id, entry in self.metadata.items():
            if entry.chunk.file_path == str_path:
                to_remove.append(id)

        if not to_remove:
            logger.info(f"No chunks found for file {str_path}")
            return

        # Mark IDs for reuse
        self._mark_ids_reusable(to_remove)

        # Update metadata
        for id in to_remove:
            if id in self.metadata:
                del self.metadata[id]

        # If more than 50% of vectors being removed, or index is small
        # We should rebuild the entire index for efficiency
        if len(to_remove) > self.index.ntotal / 2 or self.index.ntotal < 1000:
            logger.info("Rebuilding entire index due to large removal")
            self._rebuild_index()

        # Save index
        self._save_index()

        logger.info(f"Removed {len(to_remove)} chunks for file {str_path}")

    def _mark_ids_reusable(self, ids_to_remove: List[int]):
        """
        Mark IDs as reusable for future chunks.

        Args:
            ids_to_remove: List of IDs to mark as reusable
        """
        # Add the IDs to our reusable set
        for id in ids_to_remove:
            self.reusable_ids.add(id)

        logger.info(
            f"Marked {len(ids_to_remove)} IDs as reusable, total reusable: {len(self.reusable_ids)}"
        )

        # In FAISS, we can't truly remove vectors, particularly in HNSW
        # We need to mark the vectors as "removed" for our purposes
        # and reuse their IDs for new vectors
        # A real system would maintain a dynamic map from original IDs to current indices

    def _rebuild_index(self):
        """Rebuild the index with only the remaining chunks."""
        if not self.metadata:
            # If no metadata is left, create a new empty index
            self._create_index()
            self.next_id = 0
            return

        # Get all existing IDs and their corresponding chunks
        all_ids = list(self.metadata.keys())
        all_entries = [self.metadata[id] for id in all_ids]

        # Re-embed all chunks
        logger.info(f"Re-embedding {len(all_entries)} chunks for index rebuild")
        codes = [entry.chunk.code for entry in all_entries]
        embeddings = self.embedder.embed_batch(codes, batch_size=self.config.batch_size)

        # Create new index of the same type as the current one
        embedding_dim = embeddings.shape[1]

        # Create a new index with the same configuration as before
        old_index = self.index
        self._create_index()

        # Add embeddings with their original IDs
        self.index.add_with_ids(embeddings, np.array(all_ids))

        logger.info(f"Rebuilt index with {len(all_ids)} vectors")

    def update_file(self, file_path: Union[str, Path], new_chunks: List[CodeChunk]):
        """
        Update the index for a file.

        Args:
            file_path: Path to the file
            new_chunks: New chunks for the file
        """
        str_path = str(file_path)

        # Log the update details
        logger.info(
            f"Updating file in index: {str_path} with {len(new_chunks)} new chunks"
        )

        # Count existing chunks for this file
        existing_chunk_count = 0
        for id, entry in self.metadata.items():
            if entry.chunk.file_path == str_path:
                existing_chunk_count += 1

        logger.info(f"Found {existing_chunk_count} existing chunks for file {str_path}")

        # Remove old chunks for the file
        self.remove_file(file_path)

        # Verify chunks were removed
        remaining_count = 0
        for id, entry in self.metadata.items():
            if entry.chunk.file_path == str_path:
                remaining_count += 1

        logger.info(
            f"After removal: {remaining_count} chunks remain for file {str_path}"
        )

        # Add new chunks
        self.add_chunks(new_chunks)

        # Verify new chunks were added
        final_count = 0
        for id, entry in self.metadata.items():
            if entry.chunk.file_path == str_path:
                final_count += 1

        logger.info(f"After update: {final_count} chunks for file {str_path}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """
        Search for chunks semantically similar to the query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of tuples of code chunks and similarity scores
        """
        if not self.is_semantic_search_available():
            logger.warning(
                "Semantic search is not available due to missing embedding model"
            )
            return []

        if self.index is None or self.index.ntotal == 0:
            logger.warning("Empty index, no results")
            return []

        # Run garbage collection before search to avoid memory fragmentation
        gc.collect()

        try:
            # Encode the query
            query_embedding = self.embedder.embed(query)
            logger.debug(
                f"Query embedding shape: {query_embedding.shape if query_embedding is not None else 'None'}"
            )

            if query_embedding is None:
                logger.warning("Failed to encode query, no search results available")
                return []

            # Prepare the query embedding with strict memory management
            try:
                # Create a fresh copy in contiguous memory
                query_np = np.array(query_embedding, dtype=np.float32, copy=True)

                # Clean any NaN or Inf values
                if np.isnan(query_np).any() or np.isinf(query_np).any():
                    query_np = np.nan_to_num(query_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalize the query embedding
                query_np = query_np / np.linalg.norm(query_np)

                # Force contiguous memory layout in C order
                query_np = np.ascontiguousarray(query_np, dtype=np.float32)

                # Ensure correct shape (FAISS expects 2D array)
                if len(query_np.shape) == 1:
                    query_np = query_np.reshape(1, -1)

                # Ensure k is valid
                k = min(top_k, self.index.ntotal) if self.index.ntotal > 0 else top_k
                if k <= 0:
                    return []

                # Check if current metadata entries are IndexEntry objects with embeddings
                # If so, we need to use manual search method
                first_entry = (
                    next(iter(self.metadata.values())) if self.metadata else None
                )
                has_embeddings = (
                    isinstance(first_entry, IndexEntry)
                    and first_entry.embedding is not None
                )

                # Use manual search if we should not use direct search or if we have embeddings
                if (
                    hasattr(self, "use_direct_search") and not self.use_direct_search
                ) or has_embeddings:
                    return self._manual_search(query_np, k)

                try:
                    # Do another garbage collection right before search
                    gc.collect()

                    # Force memory alignment with a final contiguous array copy
                    final_query = np.ascontiguousarray(query_np, dtype=np.float32)

                    # Perform the search
                    distances, indices = self.index.search(final_query, k)
                    logger.debug(f"Search results: {len(indices[0])} indices found")
                except Exception as search_err:
                    logger.error(f"Search operation failed: {search_err}")
                    # If FAISS search fails, fallback to manual distance calculation
                    return self._manual_search(query_np, k)

                # Process the results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx == -1:  # -1 indicates padding for not enough results
                        continue

                    # Get the chunk from metadata
                    entry = self.metadata.get(int(idx))
                    if entry:
                        chunk = entry.chunk if isinstance(entry, IndexEntry) else entry
                        # Convert distance to similarity score (higher is better)
                        # Using cosine similarity since we normalized the query
                        similarity = 1.0 - float(distances[0][i]) / 2.0
                        results.append((chunk, similarity))
                        logger.debug(
                            f"Found result: {chunk.file_path} with similarity {similarity}"
                        )

                return results

            except Exception as prep_err:
                logger.error(f"Query preparation error: {prep_err}")
                return []

        except Exception as embed_err:
            logger.error(f"Embedding error: {embed_err}")
            return []

    def _manual_search(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Perform a manual search using cosine similarity.
        This is a fallback for environments where FAISS search has issues.

        Args:
            query_vector: Query embedding vector (should be 2D array)
            k: Number of results to return

        Returns:
            List of tuples of code chunks and similarity scores
        """
        try:
            # Get all chunk IDs
            chunk_ids = list(self.metadata.keys())

            if not chunk_ids:
                return []

            # We'll use stored embeddings instead of trying to reconstruct from FAISS
            all_similarities = []
            all_ids = []

            # Extract query vector to 1D for easier calculations
            q_vec = query_vector.reshape(-1)

            # For each chunk, calculate the cosine similarity
            for id_val in chunk_ids:
                entry = self.metadata.get(id_val)

                if (
                    entry
                    and hasattr(entry, "embedding")
                    and entry.embedding is not None
                ):
                    # Calculate cosine similarity using stored embedding
                    vector = entry.embedding
                    # Normalize the vector
                    vector = vector / np.linalg.norm(vector)
                    # Calculate cosine similarity
                    similarity = np.dot(vector, q_vec)
                    all_similarities.append(similarity)
                    all_ids.append(id_val)
                else:
                    pass

            if not all_similarities:
                return []

            # Convert to numpy arrays
            similarities_np = np.array(all_similarities)
            ids_np = np.array(all_ids)

            # Get the k highest similarities
            if len(similarities_np) <= k:
                top_indices = np.argsort(similarities_np)[
                    ::-1
                ]  # Reverse for descending order
            else:
                top_indices = np.argsort(similarities_np)[::-1][
                    :k
                ]  # Reverse for descending order

            # Prepare results
            results = []
            for idx in top_indices:
                chunk_id = ids_np[idx]
                similarity = similarities_np[idx]

                entry = self.metadata.get(int(chunk_id))
                if entry and hasattr(entry, "chunk"):
                    results.append((entry.chunk, float(similarity)))
                    logger.debug(
                        f"Found result: {entry.chunk.file_path} with similarity {similarity}"
                    )

            return results

        except Exception as e:
            logger.error(f"Error in manual search: {e}")
            return []


def create_semantic_index(config: CodeSleuthConfig) -> SemanticIndex:
    """
    Create a SemanticIndex from a configuration.

    Args:
        config: CodeSleuth configuration

    Returns:
        SemanticIndex instance
    """
    return SemanticIndex(config.index)


def index_repository(
    config: CodeSleuthConfig, parser: CodeParser, semantic_index: SemanticIndex
):
    """
    Index a repository.

    Args:
        config: CodeSleuth configuration
        parser: Code parser
        semantic_index: Semantic index
    """
    logger.info(f"Indexing repository: {config.repo_path}")

    # Parse the repository
    chunks = list(parser.parse_directory(config.repo_path))
    logger.info(f"Found {len(chunks)} chunks in repository")

    # Add chunks to the index
    semantic_index.add_chunks(chunks)

    logger.info(f"Indexed {len(chunks)} chunks from repository")
