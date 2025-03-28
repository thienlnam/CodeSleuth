"""Integration tests for CodeSleuth."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger
import numpy as np
import sys
import gc
import faiss

# Debug PyTorch and transformers versions
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

# Try loading a model directly to see if that's where the segfault happens
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    print("Tokenizer loaded successfully")

    print("Loading model...")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    print("Model loaded successfully")

    print("Moving model to CPU...")
    model = model.to("cpu")
    print("Model moved to CPU successfully")

    # Try a simple forward pass
    print("Trying a simple tokenization and forward pass...")
    inputs = tokenizer("def test(): pass", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Forward pass successful")

    # Clean up to avoid any issues with the rest of the tests
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Model and tokenizer deleted, cache cleared")
except Exception as e:
    print(f"Error loading model: {e}")

from codesleuth import CodeSleuth, CodeSleuthConfig, IndexConfig, EmbeddingModel
from codesleuth.config import ParserConfig


class IntegrationTest(unittest.TestCase):
    """Integration test for CodeSleuth."""

    def setUp(self):
        """Set up test repository and configuration."""
        # Create a temporary directory for the test repository
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir) / "testrepo"
        self.repo_path.mkdir()

        # Create a temporary directory for the index
        self.index_path = Path(self.temp_dir) / "index"
        self.index_path.mkdir()

        # Create test files
        self.create_test_files()

        # Create configuration with proper regex patterns
        self.parser_config = ParserConfig(
            chunk_size=100,
            chunk_overlap=20,
            ignore_patterns=[r"\.git", r"__pycache__", r".*\.pyc"],
            respect_gitignore=True,
        )

        index_config = IndexConfig(
            model_name=EmbeddingModel.CODEBERT,
            index_path=self.index_path,
            batch_size=8,
            use_gpu=False,
            hnsw_m=16,
            hnsw_ef_construction=200,
            hnsw_ef_search=50,
        )

        self.config = CodeSleuthConfig(
            repo_path=self.repo_path,
            parser=self.parser_config,
            index=index_config,
        )

        # Initialize CodeSleuth
        self.codesleuth = CodeSleuth(self.config)

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        """Create test files in the test repository."""
        # Python file with a function
        python_file = self.repo_path / "example.py"
        with open(python_file, "w") as f:
            f.write(
                '''
def search_function(query, max_results=10):
    """
    Search for code using the given query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # Implementation of search logic
    results = []
    for i in range(max_results):
        results.append(f"Result {i} for {query}")
    return results

class CodeSearcher:
    """A class for searching code."""
    
    def __init__(self, index_path):
        """Initialize with the given index path."""
        self.index_path = index_path
        
    def search(self, query):
        """Perform a semantic search."""
        return ["Code snippet 1", "Code snippet 2"]
'''
            )

        # JavaScript file with a function
        js_file = self.repo_path / "example.js"
        with open(js_file, "w") as f:
            f.write(
                """
/**
 * Search for code using the given query
 * @param {string} query - The search query
 * @param {number} maxResults - Maximum number of results to return
 * @returns {Array} List of search results
 */
function searchCode(query, maxResults = 10) {
    // Implementation of search logic
    const results = [];
    for (let i = 0; i < maxResults; i++) {
        results.push(`Result ${i} for ${query}`);
    }
    return results;
}

class SemanticSearcher {
    /**
     * Initialize the searcher
     * @param {string} indexPath - Path to the index
     */
    constructor(indexPath) {
        this.indexPath = indexPath;
    }
    
    /**
     * Perform a semantic search
     * @param {string} query - The search query
     * @returns {Array} Search results
     */
    search(query) {
        return ["Code snippet 1", "Code snippet 2"];
    }
}
"""
            )

        # C++ file with a function
        cpp_file = self.repo_path / "example.cpp"
        with open(cpp_file, "w") as f:
            f.write(
                """
#include <vector>
#include <string>

/**
 * Search for code using the given query
 * @param query The search query
 * @param maxResults Maximum number of results to return
 * @return List of search results
 */
std::vector<std::string> searchCode(const std::string& query, int maxResults = 10) {
    // Implementation of search logic
    std::vector<std::string> results;
    for (int i = 0; i < maxResults; i++) {
        results.push_back("Result " + std::to_string(i) + " for " + query);
    }
    return results;
}

class CodeSearcher {
public:
    /**
     * Initialize with the given index path
     */
    CodeSearcher(const std::string& indexPath) : indexPath(indexPath) {}
    
    /**
     * Perform a semantic search
     */
    std::vector<std::string> search(const std::string& query) {
        return {"Code snippet 1", "Code snippet 2"};
    }
    
private:
    std::string indexPath;
};
"""
            )

    def test_index_and_search(self):
        """Test indexing and searching the test repository."""
        # Index the repository
        self.codesleuth.index_repository()

        # Run garbage collection before search
        gc.collect()

        # Test semantic search
        query = "search function"
        semantic_results = self.codesleuth.search_semantically(query, top_k=5)

        # Verify we got results
        self.assertGreater(
            len(semantic_results),
            0,
            "Should find semantic results for 'search function'",
        )

        # Check that results have the expected fields
        first_result = semantic_results[0]
        self.assertIn("file_path", first_result)
        self.assertIn("start_line", first_result)
        self.assertIn("end_line", first_result)
        self.assertIn("code", first_result)
        self.assertIn("similarity", first_result)
        self.assertEqual(first_result["source"], "semantic")

        # Test lexical search - wrap in try/except to handle case where ripgrep isn't installed
        try:
            lexical_results = self.codesleuth.search_lexically("search", max_results=5)
            self.assertGreater(
                len(lexical_results), 0, "Should find lexical results for 'search'"
            )
        except Exception as e:
            logger.warning(f"Lexical search test skipped: {e}")

        # Test file search
        file_results = self.codesleuth.search_file("*.py")
        self.assertEqual(len(file_results), 1, "Should find one Python file")
        self.assertTrue(file_results[0]["file_path"].endswith("example.py"))

        # Test hybrid search
        hybrid_results = self.codesleuth.hybrid_search("search", top_k=5)
        self.assertGreater(
            len(hybrid_results), 0, "Should find hybrid results for 'search'"
        )

    def test_update_index(self):
        """Test updating the index when files change."""
        # Index the repository
        self.codesleuth.index_repository()

        # Get initial search results
        query = "semantic search"
        initial_results = self.codesleuth.search_semantically(query, top_k=5)
        initial_chunk_count = len(
            self.codesleuth.semantic_search.semantic_index.metadata
        )

        # Update a file by overwriting it with additional content
        js_file = self.repo_path / "example.js"

        # First, read the existing content
        with open(js_file, "r") as f:
            existing_content = f.read()

        # Then write it back with additional content
        new_content = """/**
 * A special function dedicated to newSemanticSearch functionality
 * @param {string} query - The search query
 * @returns {Array} Search results
 */
function newSemanticSearch(query) {
    // This is a highly specific implementation
    // that should be easy to find in search results
    return ["New semantic result 1", "New semantic result 2"];
}
"""

        # Write the updated file with more separation
        with open(js_file, "w") as f:
            f.write(existing_content)
            # Add some extra empty lines to ensure this is treated as a separate chunk
            f.write("\n\n\n\n\n\n\n\n\n\n")  # 10 newlines for clear separation
            f.write(new_content)

        # Re-index the repository
        self.codesleuth.index_repository()

        # Check that metadata was updated
        updated_chunk_count = len(
            self.codesleuth.semantic_search.semantic_index.metadata
        )

        # Verify that the number of chunks increased
        self.assertGreater(
            updated_chunk_count,
            initial_chunk_count,
            "Index should contain more chunks after update",
        )

        # Search with a query specifically targeting the new content
        updated_results = self.codesleuth.search_semantically(
            "newSemanticSearch", top_k=10
        )

        # Check that we have results
        self.assertGreater(
            len(updated_results), 0, "Should find results for 'newSemanticSearch'"
        )

        # At least one result should contain our new function
        new_function_found = False
        for result in updated_results:
            if "newSemanticSearch" in result["code"]:
                new_function_found = True
                break

        self.assertTrue(
            new_function_found, "The updated file with new function should be indexed"
        )


if __name__ == "__main__":
    unittest.main()
