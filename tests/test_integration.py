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
        print("DEBUG: Starting indexing of repository", file=sys.stderr)
        self.codesleuth.index_repository()
        print("DEBUG: Indexing completed successfully", file=sys.stderr)

        # Print index details
        print(
            f"DEBUG: Index has {self.codesleuth.semantic_search.semantic_index.index.ntotal if hasattr(self.codesleuth.semantic_search.semantic_index.index, 'ntotal') else 'unknown'} vectors",
            file=sys.stderr,
        )

        # Force garbage collection before search
        print("DEBUG: Running garbage collection before search", file=sys.stderr)
        gc.collect()

        try:
            # Get access to the internal index
            index = self.codesleuth.semantic_search.semantic_index.index
            index_type = type(index).__name__
            print(f"DEBUG: Index type: {index_type}", file=sys.stderr)

            # Check if it's an IDMap-wrapped HNSW
            is_hnsw = False
            if hasattr(index, "index") and isinstance(index, faiss.IndexIDMap):
                inner_index = index.index
                inner_type = type(inner_index).__name__
                print(f"DEBUG: Inner index type: {inner_type}", file=sys.stderr)
                if isinstance(inner_index, faiss.IndexHNSWFlat):
                    is_hnsw = True
                    print(
                        f"DEBUG: HNSW parameters: M={inner_index.hnsw.M}, efConstruction={inner_index.hnsw.efConstruction}, efSearch={inner_index.hnsw.efSearch}",
                        file=sys.stderr,
                    )

            # Test semantic search
            print("DEBUG: About to run semantic search", file=sys.stderr)
            query = "search function"

            # Perform the semantic search through the API
            print(f"DEBUG: Searching with query: '{query}', top_k=5", file=sys.stderr)
            semantic_results = self.codesleuth.search_semantically(query, top_k=5)
            print(
                f"DEBUG: Semantic search returned {len(semantic_results)} results",
                file=sys.stderr,
            )

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

        except Exception as e:
            print(f"DEBUG: Error in semantic search: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            raise

        # Test lexical search - wrap in try/except to handle case where ripgrep isn't installed
        try:
            lexical_results = self.codesleuth.search_lexically("search", max_results=5)
            self.assertGreater(
                len(lexical_results), 0, "Should find lexical results for 'search'"
            )
        except Exception as e:
            print(f"Lexical search test skipped: {e}")

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
        print(f"DEBUG: Initial chunk count: {initial_chunk_count}")

        # Print initial results for debugging
        print("DEBUG: Initial search results:")
        for i, result in enumerate(initial_results):
            print(
                f"DEBUG: Result {i}: {result['file_path']} - {result['code'][:50]}..."
            )

        # Update a file by overwriting it with additional content
        js_file = self.repo_path / "example.js"
        print(f"DEBUG: Updating file: {js_file}")

        # First, read the existing content
        with open(js_file, "r") as f:
            existing_content = f.read()
            print(
                f"DEBUG: Original JS file has {len(existing_content)} chars and {existing_content.count('\\n')} lines"
            )

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

        # Verify the file was updated
        with open(js_file, "r") as f:
            content = f.read()
            print(
                f"DEBUG: Updated JS file has {len(content)} chars and {content.count('\\n')} lines"
            )
            print(
                f"DEBUG: Updated file now contains newSemanticSearch: {'newSemanticSearch' in content}"
            )
            print(f"DEBUG: Updated file content (last 300 chars): {content[-300:]}")

        # Debug the parser's chunking directly
        print("DEBUG: Testing parser directly on updated file:")
        # Create a parser directly since it's not exposed through the API
        from codesleuth.indexing.parser import CodeParser

        parser = CodeParser(self.config)
        chunks = list(parser.parse_file(js_file))
        print(f"DEBUG: Parser found {len(chunks)} chunks in updated JS file")
        for i, chunk in enumerate(chunks):
            print(
                f"DEBUG: Chunk {i}: lines {chunk.start_line}-{chunk.end_line}, code snippet: {chunk.code[:80].replace(chr(10), ' ')}..."
            )
            print(
                f"DEBUG: Contains newSemanticSearch: {'newSemanticSearch' in chunk.code}"
            )
            if "newSemanticSearch" in chunk.code:
                print(f"DEBUG: Full chunk with newSemanticSearch: {chunk.code}")

        # Also examine chunk_size and chunk_overlap in the parser config
        print(
            f"DEBUG: Chunk size: {self.config.parser.chunk_size}, Chunk overlap: {self.config.parser.chunk_overlap}"
        )

        # Re-index the repository
        print("DEBUG: Re-indexing repository")
        self.codesleuth.index_repository()

        # Check that metadata was updated - this is a good sign that indexing worked
        updated_chunk_count = len(
            self.codesleuth.semantic_search.semantic_index.metadata
        )
        print(f"DEBUG: Updated chunk count: {updated_chunk_count}")

        # Verify that the number of chunks increased
        self.assertGreater(
            updated_chunk_count,
            initial_chunk_count,
            "Index should contain more chunks after update",
        )

        # Search with a query specifically targeting the new content
        print("DEBUG: Searching for 'newSemanticSearch'")
        updated_results = self.codesleuth.search_semantically(
            "newSemanticSearch", top_k=10
        )
        print(f"DEBUG: Found {len(updated_results)} results")

        # Print all results for debugging
        for i, result in enumerate(updated_results):
            print(
                f"DEBUG: Result {i}: {result['file_path']} - {result['code'][:50]}..."
            )
            print(f"DEBUG: Full code for result {i}: {result['code']}")
            print(
                f"DEBUG: Contains newSemanticSearch: {'newSemanticSearch' in result['code']}"
            )

        # Check that we have results
        self.assertGreater(
            len(updated_results), 0, "Should find results for 'newSemanticSearch'"
        )

        # At least one result should contain our new function
        new_function_found = False
        for result in updated_results:
            print(f"DEBUG: Checking result: {'newSemanticSearch' in result['code']}")
            if "newSemanticSearch" in result["code"]:
                new_function_found = True
                break

        # Special test to directly search the index with the exact new function name
        direct_results = self.codesleuth.search_semantically(
            "newSemanticSearch function special", top_k=10
        )
        print(f"DEBUG: Direct search found {len(direct_results)} results")
        for i, result in enumerate(direct_results):
            print(f"DEBUG: Direct result {i}: {result['code'][:50]}...")
            print(
                f"DEBUG: Contains newSemanticSearch: {'newSemanticSearch' in result['code']}"
            )

        self.assertTrue(
            new_function_found, "The updated file with new function should be indexed"
        )


if __name__ == "__main__":
    unittest.main()
