"""
Semantic index module for CodeSleuth.

This module handles the embedding of code chunks and indexing them in a vector database.
It supports incremental updates to the index when code files change.
"""

import os
import pickle
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
    EmbeddingModel.DISTILCODEBERT: "microsoft/codebert-base",
    # We can add more models here as needed
}


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
        """
        # Tokenize the code
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings[0]

    def embed_batch(self, codes: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of code snippets.

        Args:
            codes: List of code snippets to embed
            batch_size: Batch size for embedding computation

        Returns:
            Array of embedding vectors
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

        # Initialize the embedder
        self.embedder = CodeEmbedder(
            model_name=config.model_name,
            use_gpu=config.use_gpu,
        )

        # Initialize FAISS index
        self.index = None
        self.metadata: Dict[int, CodeChunk] = {}
        self.next_id = 0
        self.reusable_ids = set()  # Track IDs that can be reused

        # Load index if it exists
        if self._index_exists():
            self._load_index()
        else:
            self._create_index()

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

        # Create FAISS index
        # By default, use a flat L2 index for simplicity and accuracy
        # For larger repositories, use HNSW index for better performance
        embedding_dim = 768  # CodeBERT embedding dimension

        if self.config.use_hnsw:
            # Create an HNSW index for better search performance
            # Parameters from configuration
            M = self.config.hnsw_m
            efConstruction = self.config.hnsw_ef_construction

            # Create the HNSW index
            hnsw_index = faiss.IndexHNSWFlat(embedding_dim, M, faiss.METRIC_L2)
            hnsw_index.hnsw.efConstruction = efConstruction
            hnsw_index.hnsw.efSearch = self.config.hnsw_ef_search

            # Wrap with IndexIDMap to support add_with_ids
            self.index = faiss.IndexIDMap(hnsw_index)

            logger.info(
                f"Created new HNSW index with dimension {embedding_dim}, M={M}, efConstruction={efConstruction}"
            )
        else:
            # Use a simple flat index for smaller repositories
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"Created new flat index with dimension {embedding_dim}")

    def _save_index(self):
        """Save the index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save metadata and reusable IDs
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump((self.metadata, self.next_id, self.reusable_ids), f)

        logger.info(
            f"Saved index with {self.index.ntotal} vectors to {self.index_path}"
        )

    def _load_index(self):
        """Load the index and metadata from disk."""
        # Load FAISS index
        index_file = self.index_path / "index.faiss"
        self.index = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, "rb") as f:
            try:
                # Try to load with reusable_ids (newer version)
                self.metadata, self.next_id, self.reusable_ids = pickle.load(f)
            except ValueError:
                # Fall back to loading without reusable_ids (older version)
                self.metadata, self.next_id = pickle.load(f)
                self.reusable_ids = set()

        # Configure HNSW search parameters if using HNSW index
        if self.config.use_hnsw:
            # The wrapped index is accessed through .index if this is an IndexIDMap
            if isinstance(self.index, faiss.IndexIDMap):
                if isinstance(self.index.index, faiss.IndexHNSWFlat):
                    self.index.index.hnsw.efSearch = self.config.hnsw_ef_search
            # Direct HNSW index
            elif isinstance(self.index, faiss.IndexHNSWFlat):
                self.index.hnsw.efSearch = self.config.hnsw_ef_search

        logger.info(
            f"Loaded index with {self.index.ntotal} vectors from {self.index_path}"
        )
        logger.info(
            f"Metadata contains {len(self.metadata)} entries, next_id={self.next_id}, {len(self.reusable_ids)} reusable IDs"
        )

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

        # Extract code snippets
        codes = [chunk.code for chunk in chunks]

        # Compute embeddings in batches
        logger.info(f"Computing embeddings for {len(codes)} chunks")
        embeddings = self.embedder.embed_batch(codes, batch_size=self.config.batch_size)

        # Assign IDs, preferring reusable IDs when available
        ids = []
        id_to_chunk = {}

        for i, chunk in enumerate(chunks):
            chunk_id = self._get_next_id()
            ids.append(chunk_id)
            id_to_chunk[chunk_id] = chunk

            # Only increment next_id if we used it
            if chunk_id == self.next_id:
                self.next_id += 1

        # Add to index
        self.index.add_with_ids(embeddings, np.array(ids))

        # Update metadata
        self.metadata.update(id_to_chunk)

        # Save index
        self._save_index()

        logger.info(f"Added {len(chunks)} chunks to index")

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
        # Remove old chunks for the file
        self.remove_file(file_path)

        # Add new chunks
        self.add_chunks(new_chunks)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """
        Search the index for similar code chunks.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Empty index, no results")
            return []

        # Embed the query
        query_embedding = self.embedder.embed(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert to results
        results = []
        for i, idx in enumerate(indices[0]):
            if (
                idx == -1
            ):  # FAISS returns -1 for padding when there are not enough results
                continue

            chunk = self.metadata.get(int(idx))
            if chunk:
                similarity = 1.0 / (
                    1.0 + distances[0][i]
                )  # Convert distance to similarity score
                results.append((chunk, similarity))

        return results


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
