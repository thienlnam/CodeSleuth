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
import traceback
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

# Model names for HuggingFace
MODEL_MAPPING = {
    EmbeddingModel.CODEBERT: "microsoft/codebert-base",
    # We can add more models here as needed
}


# Add GPU support detection
def get_gpu_resources():
    """
    Get available GPU resources for FAISS.

    Returns:
        Tuple of (bool, List): (GPU available, GPU resources list)
    """
    try:
        # Check if faiss-gpu is installed
        import importlib.util

        has_gpu_support = importlib.util.find_spec("faiss.gpu") is not None

        # Check if CUDA is available through PyTorch
        cuda_available = torch.cuda.is_available()

        if has_gpu_support and cuda_available:
            logger.info(
                f"FAISS GPU support detected, {torch.cuda.device_count()} CUDA devices available"
            )
            # Get available GPU resources
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                gpu_resources.append(res)
            return True, gpu_resources
        else:
            if not has_gpu_support:
                logger.info(
                    "FAISS GPU support not installed (faiss-gpu package required)"
                )
            if not cuda_available:
                logger.info("CUDA not available for PyTorch")
            return False, []
    except Exception as e:
        logger.warning(f"Error detecting GPU resources: {e}")
        return False, []


@dataclass
class IndexEntry:
    """Entry in the semantic index."""

    id: int
    chunk: CodeChunk
    embedding: Optional[np.ndarray] = None


class CodeEmbedder:
    """Code embedding model."""

    def __init__(self, model_name: str, use_gpu: bool = False):
        """
        Initialize the code embedder.

        Args:
            model_name: Name of the model to use
            use_gpu: Whether to use GPU for embedding computation
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        model_hf_name = MODEL_MAPPING.get(model_name, model_name)
        logger.info(f"Loading model: {model_hf_name}")

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_hf_name)
        self.model = AutoModel.from_pretrained(model_hf_name).to(self.device)

        # Set model to evaluation mode
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
        import sys

        print(f"EMBED DEBUG: Starting embed of code: {code[:20]}...", file=sys.stderr)

        try:
            # Tokenize the code
            print(f"EMBED DEBUG: Tokenizing code", file=sys.stderr)
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            print(f"EMBED DEBUG: Code tokenized successfully", file=sys.stderr)

            try:
                print(
                    f"EMBED DEBUG: Moving inputs to device {self.device}",
                    file=sys.stderr,
                )
                inputs = inputs.to(self.device)
                print(
                    f"EMBED DEBUG: Inputs moved to device successfully", file=sys.stderr
                )
            except Exception as e:
                print(
                    f"EMBED DEBUG: Error moving inputs to device: {e}", file=sys.stderr
                )
                raise

            # Compute embeddings
            print(f"EMBED DEBUG: Computing embeddings", file=sys.stderr)
            with torch.no_grad():
                try:
                    print(f"EMBED DEBUG: Running model forward pass", file=sys.stderr)
                    outputs = self.model(**inputs)
                    print(f"EMBED DEBUG: Forward pass successful", file=sys.stderr)
                except Exception as e:
                    print(
                        f"EMBED DEBUG: Error in model forward pass: {e}",
                        file=sys.stderr,
                    )
                    raise

                try:
                    print(f"EMBED DEBUG: Extracting embeddings", file=sys.stderr)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    print(
                        f"EMBED DEBUG: Embeddings extracted successfully, shape: {embeddings.shape}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"EMBED DEBUG: Error extracting embeddings: {e}",
                        file=sys.stderr,
                    )
                    raise

            return embeddings[0]
        except Exception as e:
            print(f"EMBED DEBUG: Uncaught error in embed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
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
        all_embeddings = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i : i + batch_size]

            # Tokenize the batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
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

        # Check for GPU support
        self.has_gpu, self.gpu_resources = get_gpu_resources()
        # Override config setting if GPU is not available
        self.use_gpu = config.use_gpu and self.has_gpu

        # Initialize the embedder
        try:
            self.embedder = CodeEmbedder(
                model_name=config.model_name,
                use_gpu=self.use_gpu,
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

        # Create FAISS index with HNSW for better performance
        embedding_dim = 768  # CodeBERT embedding dimension

        # Detect Python 3.13 on M1/M2 Macs for compatibility issues
        is_python_3_13 = sys.version_info.major == 3 and sys.version_info.minor == 13
        is_apple_silicon = platform.processor() == "arm"
        problematic_environment = is_python_3_13 and is_apple_silicon

        try:
            if problematic_environment:
                # Known issue with HNSW on Python 3.13 + Apple Silicon
                logger.info(
                    "Detected Python 3.13 on M1/M2 Mac, using FlatL2 index for compatibility"
                )
                flat_index = faiss.IndexFlatL2(embedding_dim)
                self.index = faiss.IndexIDMap(flat_index)
            else:
                # Try to create HNSW index for better performance
                logger.info(
                    f"Creating HNSW index with M={self.config.hnsw_m}, efConstruction={self.config.hnsw_ef_construction}"
                )

                # Step 1: Create the HNSW index with the specified parameters
                hnsw_index = faiss.IndexHNSWFlat(
                    embedding_dim, self.config.hnsw_m  # Number of connections per layer
                )

                # Step 2: Set the HNSW construction-time and search-time parameters
                hnsw_index.hnsw.efConstruction = self.config.hnsw_ef_construction
                hnsw_index.hnsw.efSearch = self.config.hnsw_ef_search

                # Step 3: Wrap with IDMap to support add_with_ids
                self.index = faiss.IndexIDMap(hnsw_index)

                logger.info(f"Created HNSW index with dimension {embedding_dim}")

            # Log the type information
            logger.info(f"Created index of type: {type(self.index).__name__}")
            if hasattr(self.index, "index"):
                logger.info(f"Inner index type: {type(self.index.index).__name__}")

                # Log HNSW parameters if applicable
                inner_index = self.index.index
                if hasattr(inner_index, "hnsw"):
                    logger.info(
                        f"HNSW parameters: M={inner_index.hnsw.M}, "
                        f"efConstruction={inner_index.hnsw.efConstruction}, "
                        f"efSearch={inner_index.hnsw.efSearch}"
                    )
        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}, using FlatL2 as fallback")
            # Fallback to FlatL2 if creation fails
            flat_index = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIDMap(flat_index)
            logger.info("Created FlatL2 index as fallback")

        # Use a more robust search method for problematic environments
        self.use_direct_search = not problematic_environment

    def _save_index(self):
        """Save the index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        # If index is on GPU, move it back to CPU for saving
        index_to_save = self.index
        if self.use_gpu and self.gpu_resources:
            try:
                logger.info("Moving index from GPU to CPU for saving")
                index_to_save = faiss.index_gpu_to_cpu(self.index)
                logger.info("Index successfully moved to CPU")
            except Exception as e:
                logger.warning(
                    f"Failed to move index to CPU for saving: {e}. Using current index."
                )
                index_to_save = self.index

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(index_to_save, str(index_file))

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
            cpu_index = faiss.read_index(str(index_file))

            # Log index type information
            logger.info(f"Loaded index type: {type(cpu_index).__name__}")
            if hasattr(cpu_index, "index"):
                inner_index = cpu_index.index
                logger.info(f"Loaded inner index type: {type(inner_index).__name__}")

                # Check if inner index is HNSW and set parameters
                if hasattr(inner_index, "hnsw"):
                    logger.info(
                        f"HNSW index loaded with efSearch={inner_index.hnsw.efSearch}, M={inner_index.hnsw.M}"
                    )
                    # Ensure efSearch is set according to config
                    inner_index.hnsw.efSearch = self.config.hnsw_ef_search
                    logger.info(
                        f"Updated HNSW efSearch to {self.config.hnsw_ef_search}"
                    )

            # Move to GPU if available and requested
            if self.use_gpu and self.gpu_resources:
                try:
                    logger.info(f"Moving loaded index to GPU 0")
                    gpu_index = faiss.index_cpu_to_gpu(
                        self.gpu_resources[0], 0, cpu_index
                    )
                    self.index = gpu_index
                    logger.info("Index successfully moved to GPU")
                except Exception as e:
                    logger.warning(
                        f"Failed to move index to GPU: {e}. Using CPU index."
                    )
                    self.index = cpu_index
            else:
                self.index = cpu_index

            # Load metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, "rb") as f:
                loaded_data = pickle.load(f)

                # Handle different metadata formats
                if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
                    # Previous format: (metadata_dict, next_id, reusable_ids)
                    metadata_dict, self.next_id, self.reusable_ids = loaded_data

                    # Convert old metadata format to new IndexEntry format if needed
                    self.metadata = {}
                    for id_val, item in metadata_dict.items():
                        if isinstance(item, IndexEntry):
                            # Already in the right format
                            self.metadata[id_val] = item
                        else:
                            # Convert to IndexEntry
                            self.metadata[id_val] = IndexEntry(
                                id=id_val,
                                chunk=item,
                                embedding=None,  # We don't have the embedding for old entries
                            )
                else:
                    # Unexpected format, use as is but log a warning
                    self.metadata = loaded_data
                    logger.warning(
                        f"Loaded metadata in unexpected format: {type(loaded_data)}"
                    )
                    self.next_id = max(self.metadata.keys()) + 1 if self.metadata else 0
                    self.reusable_ids = set()

            # Check if any index entry has embeddings
            has_embeddings = any(
                isinstance(entry, IndexEntry) and entry.embedding is not None
                for entry in self.metadata.values()
            )

            if has_embeddings:
                logger.info("Loaded metadata contains embeddings for manual search")
            else:
                logger.warning(
                    "Loaded metadata does not contain embeddings, retrieval may be limited"
                )

            # Detect Python 3.13 on M1/M2 Macs for compatibility issues
            is_python_3_13 = (
                sys.version_info.major == 3 and sys.version_info.minor == 13
            )
            is_apple_silicon = platform.processor() == "arm"
            self.use_direct_search = not (is_python_3_13 and is_apple_silicon)

            logger.info(
                f"Loaded index with {self.index.ntotal} vectors from {self.index_path}"
            )
            logger.info(
                f"Metadata contains {len(self.metadata)} entries, next_id={self.next_id}, {len(self.reusable_ids)} reusable IDs"
            )
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Create a new index as fallback
            self._create_index()

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
        Add code chunks to the index.

        Args:
            chunks: List of code chunks to add
        """
        if not chunks:
            return

        if not self.is_semantic_search_available():
            logger.warning("Embedding model not available - skipping semantic indexing")
            return

        try:
            # Extract code snippets
            codes = [chunk.code for chunk in chunks]

            # Compute embeddings in batches
            logger.info(f"Computing embeddings for {len(codes)} chunks")
            embeddings = self.embedder.embed_batch(
                codes, batch_size=self.config.batch_size
            )

            # Assign IDs, preferring reusable IDs when available
            ids = []
            id_to_chunk = {}
            id_to_embedding = {}

            for i, chunk in enumerate(chunks):
                chunk_id = self._get_next_id()
                ids.append(chunk_id)
                id_to_chunk[chunk_id] = chunk

                # Store the embedding for the chunk
                if i < len(embeddings):
                    id_to_embedding[chunk_id] = embeddings[i]

                # Only increment next_id if we used it
                if chunk_id == self.next_id:
                    self.next_id += 1

            # Add to index
            self.index.add_with_ids(embeddings, np.array(ids))

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
            logger.error(f"Failed to add chunks to index: {e}")
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
        for id, chunk in self.metadata.items():
            if chunk.file_path == str_path:
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
        all_chunks = [self.metadata[id] for id in all_ids]

        # Re-embed all chunks
        logger.info(f"Re-embedding {len(all_chunks)} chunks for index rebuild")
        codes = [chunk.code for chunk in all_chunks]
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
            if isinstance(entry, IndexEntry):
                chunk = entry.chunk
            else:
                chunk = entry

            if chunk.file_path == str_path:
                existing_chunk_count += 1

        logger.info(f"Found {existing_chunk_count} existing chunks for file {str_path}")

        # Remove old chunks for the file
        self.remove_file(file_path)

        # Verify chunks were removed
        remaining_count = 0
        for id, entry in self.metadata.items():
            if isinstance(entry, IndexEntry):
                chunk = entry.chunk
            else:
                chunk = entry

            if chunk.file_path == str_path:
                remaining_count += 1

        logger.info(
            f"After removal: {remaining_count} chunks remain for file {str_path}"
        )

        # Add new chunks
        self.add_chunks(new_chunks)

        # Verify new chunks were added
        final_count = 0
        for id, entry in self.metadata.items():
            if isinstance(entry, IndexEntry):
                chunk = entry.chunk
            else:
                chunk = entry

            if chunk.file_path == str_path:
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
        import sys

        print(
            f"DEBUG: Starting search for query: '{query}', top_k={top_k}",
            file=sys.stderr,
        )

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
            print(f"DEBUG: Encoding query with model", file=sys.stderr)
            query_embedding = self.embedder.embed(query)

            print(
                f"DEBUG: Query encoded, shape={np.shape(query_embedding) if query_embedding is not None else 'None'}",
                file=sys.stderr,
            )

            if query_embedding is None:
                logger.warning("Failed to encode query, no search results available")
                return []

            # Prepare the query embedding with strict memory management
            try:
                # Step 1: Create a fresh copy in contiguous memory
                print(f"DEBUG: Preparing query embedding", file=sys.stderr)
                query_np = np.array(query_embedding, dtype=np.float32, copy=True)

                # Step 2: Clean any NaN or Inf values
                if np.isnan(query_np).any() or np.isinf(query_np).any():
                    query_np = np.nan_to_num(query_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Step 3: Force contiguous memory layout in C order
                query_np = np.ascontiguousarray(query_np, dtype=np.float32)

                # Step 4: Ensure correct shape (FAISS expects 2D array)
                if len(query_np.shape) == 1:
                    query_np = query_np.reshape(1, -1)

                print(
                    f"DEBUG: Query prepared, shape={query_np.shape}, dtype={query_np.dtype}, contiguous={query_np.flags.c_contiguous}",
                    file=sys.stderr,
                )

                # Step 5: Log index information
                print(
                    f"DEBUG: Index type: {type(self.index).__name__}", file=sys.stderr
                )
                print(
                    f"DEBUG: Index size: {self.index.ntotal} vectors", file=sys.stderr
                )

                if hasattr(self.index, "index"):
                    inner_index = self.index.index
                    print(
                        f"DEBUG: Inner index type: {type(inner_index).__name__}",
                        file=sys.stderr,
                    )

                    if hasattr(inner_index, "hnsw"):
                        print(
                            f"DEBUG: HNSW parameters: M={inner_index.hnsw.M}, efSearch={inner_index.hnsw.efSearch}",
                            file=sys.stderr,
                        )

                # Step 6: Ensure k is valid
                k = min(top_k, self.index.ntotal) if self.index.ntotal > 0 else top_k
                if k <= 0:
                    print(
                        f"DEBUG: Invalid k={k}, returning empty results",
                        file=sys.stderr,
                    )
                    return []

                print(f"DEBUG: Searching for top {k} results", file=sys.stderr)

                # Step 7: Check if current metadata entries are IndexEntry objects with embeddings
                # If not, we need to use direct FAISS search
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
                    print(f"DEBUG: Using manual distance calculation", file=sys.stderr)
                    return self._manual_search(query_np, k)

                try:
                    # Do another garbage collection right before search
                    gc.collect()

                    # Perform the search safely - this is where the segmentation fault can happen
                    print(f"DEBUG: Executing FAISS search", file=sys.stderr)
                    # Force memory alignment with a final contiguous array copy
                    final_query = np.ascontiguousarray(query_np, dtype=np.float32)

                    distances, indices = self.index.search(final_query, k)

                    print(
                        f"DEBUG: Search completed successfully with {len(indices[0])} results",
                        file=sys.stderr,
                    )
                    print(
                        f"DEBUG: Distances: {distances[0][:5]}, Indices: {indices[0][:5]}",
                        file=sys.stderr,
                    )
                except Exception as search_err:
                    print(
                        f"DEBUG: Search failed with error: {search_err}",
                        file=sys.stderr,
                    )
                    logger.error(f"Search operation failed: {search_err}")
                    traceback.print_exc(file=sys.stderr)

                    # If FAISS search fails, fallback to manual distance calculation
                    print(
                        f"DEBUG: Falling back to manual distance calculation",
                        file=sys.stderr,
                    )
                    return self._manual_search(query_np, k)

                # Step 8: Process the results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx == -1:  # -1 indicates padding for not enough results
                        continue

                    # Get the chunk from metadata
                    entry = self.metadata.get(int(idx))
                    if entry:
                        chunk = entry.chunk if isinstance(entry, IndexEntry) else entry
                        # Convert distance to similarity score (higher is better)
                        similarity = 1.0 / (1.0 + float(distances[0][i]))
                        results.append((chunk, similarity))

                print(
                    f"DEBUG: Returning {len(results)} processed results",
                    file=sys.stderr,
                )
                return results

            except Exception as prep_err:
                print(
                    f"DEBUG: Error preparing query or processing results: {prep_err}",
                    file=sys.stderr,
                )
                logger.error(f"Query preparation error: {prep_err}")
                traceback.print_exc(file=sys.stderr)
                return []

        except Exception as embed_err:
            print(f"DEBUG: Error during query embedding: {embed_err}", file=sys.stderr)
            logger.error(f"Embedding error: {embed_err}")
            traceback.print_exc(file=sys.stderr)
            return []

    def _manual_search(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Perform a manual search using L2 distance calculation.
        This is a fallback for environments where FAISS search has issues.

        Args:
            query_vector: Query embedding vector (should be 2D array)
            k: Number of results to return

        Returns:
            List of tuples of code chunks and similarity scores
        """
        print(f"DEBUG: Starting manual L2 distance calculation", file=sys.stderr)

        try:
            # Get all chunk IDs
            chunk_ids = list(self.metadata.keys())

            if not chunk_ids:
                print(f"DEBUG: No chunks in metadata", file=sys.stderr)
                return []

            # We'll use stored embeddings instead of trying to reconstruct from FAISS
            all_distances = []
            all_ids = []

            # Extract query vector to 1D for easier calculations
            q_vec = query_vector.reshape(-1)

            print(
                f"DEBUG: Calculating distances for {len(chunk_ids)} vectors",
                file=sys.stderr,
            )

            # For each chunk, calculate the L2 distance
            for id_val in chunk_ids:
                entry = self.metadata.get(id_val)

                if (
                    entry
                    and hasattr(entry, "embedding")
                    and entry.embedding is not None
                ):
                    # Calculate L2 distance using stored embedding
                    vector = entry.embedding
                    dist = np.sum((vector - q_vec) ** 2)

                    all_distances.append(dist)
                    all_ids.append(id_val)
                else:
                    print(
                        f"DEBUG: No stored embedding for ID {id_val}", file=sys.stderr
                    )

            if not all_distances:
                print(f"DEBUG: No valid distances calculated", file=sys.stderr)
                return []

            # Convert to numpy arrays
            distances_np = np.array(all_distances)
            ids_np = np.array(all_ids)

            # Get the k smallest distances
            if len(distances_np) <= k:
                top_indices = np.argsort(distances_np)
            else:
                top_indices = np.argsort(distances_np)[:k]

            print(f"DEBUG: Found {len(top_indices)} top results", file=sys.stderr)

            # Prepare results
            results = []
            for idx in top_indices:
                chunk_id = ids_np[idx]
                distance = distances_np[idx]

                entry = self.metadata.get(int(chunk_id))
                if entry and hasattr(entry, "chunk"):
                    # Convert distance to similarity score (higher is better)
                    similarity = 1.0 / (1.0 + float(distance))
                    results.append((entry.chunk, similarity))

            print(
                f"DEBUG: Returning {len(results)} manual search results",
                file=sys.stderr,
            )
            return results

        except Exception as e:
            print(f"DEBUG: Error in manual search: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
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
