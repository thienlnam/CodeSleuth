# CodeSleuth

CodeSleuth is a **local-first code search and retrieval engine** designed specifically for developers working with **large codebases** and **Large Language Models (LLMs)**. It indexes your code locally, creates semantic embeddings, and provides fast, context-aware search without relying on external cloud services, keeping your code private and accessible offline.

## Why CodeSleuth?

Traditional code search tools often fall short when dealing with the complexities of modern development:

-   **Standard `grep` or IDE search:** Struggle with semantic understanding ("find code _like_ this," not just _containing_ this text).
-   **Cloud-based solutions:** Require uploading your code, raising privacy concerns and introducing external dependencies.
-   **LLM Integration:** Finding the _right_ code context to feed an LLM for tasks like code generation, explanation, or debugging can be challenging and time-consuming.

CodeSleuth addresses these gaps by offering:

-   **Privacy & Control:** Your code stays on your machine. Period.
-   **Semantic Power:** Understands the _meaning_ of your code and queries, finding relevant snippets even if the keywords don't match exactly.
-   **LLM-Ready Context:** Retrieves focused, relevant code chunks perfect for augmenting LLM prompts.
-   **Speed on Large Repositories:** Optimized indexing and search architecture (including MLX for Apple Silicon) handle large amounts of code efficiently.

## Key Features

-   **🚀 Local-First Architecture**:
    -   All indexing and searching happen on your machine.
    -   No cloud dependencies, API keys, or data sharing required.
    -   Works fully offline.
-   **🧠 LLM-Optimized Search**:
    -   **Semantic Search:** Find code based on natural language descriptions or conceptual similarity.
    -   **Context-Aware Results:** Retrieves meaningful code chunks (often function/class level) ideal for LLM context.
-   **⚡ High Performance**:
    -   **MLX Acceleration:** Native embedding calculation support for Apple Silicon (M1/M2/M3) via `mlx-embedding-models`. Requires installing optional dependencies (`pip install codesleuth[arm]`).
    -   **Efficient Indexing:** Uses optimized FAISS vector indexes (`HNSW` or `FlatL2`) for fast lookups.
    -   **Fast Code Parsing:** Leverages Tree-sitter for quick and accurate code analysis.
    -   **Optimized for Large Codebases:** Efficient chunking, batch processing, and performant search algorithms.
-   **🔍 Comprehensive Search Capabilities**:
    -   **Semantic Search:** Find code by meaning.
    *   **Lexical Search:** Fast, precise text/regex search powered by `ripgrep`. _(Note: `ripgrep` needs to be installed on your system)_.
-   **🌐 Broad Language Support**:
    -   Utilizes Tree-sitter for parsing, supporting a wide range of languages out-of-the-box (Python, JS/TS, Java, C/C++, Go, Rust, PHP). See Tree-sitter documentation for parsers.

## Installation

```bash
# Standard installation
pip install codesleuth

# To enable MLX acceleration on Apple Silicon (ARM), install optional dependencies:
pip install codesleuth[arm]
# Or using Poetry:
# poetry install --with arm

# Ensure ripgrep is installed for Lexical Search
# (e.g., brew install ripgrep, apt install ripgrep)
# sudo apt-get install ripgrep # Ubuntu Example
# brew install ripgrep # MacOS Example
```

## Quick Start

```python
from codesleuth import CodeSleuth
from codesleuth.config import CodeSleuthConfig, EmbeddingModel

# 1. Initialize CodeSleuth with your repository path
#    (Ensure the path points to the root of your code repository)
config = CodeSleuthConfig(repo_path="/path/to/your/repo")
codesleuth = CodeSleuth(config)

# 2. Index your repository
#    (This creates embeddings and builds the search index locally)
#    (Uses MLX automatically on Apple Silicon if available and enabled)
print("Starting repository indexing...")
codesleuth.index_repository()
print("Indexing complete.")

# 3. Perform a semantic search
#    (Find code based on meaning/description)
query = "authentication service implementation"
print(f"Performing semantic search for: '{query}'")
if codesleuth.is_semantic_search_available():
    # Search for code semantically
    results = codesleuth.search_semantically( # Use the main search method
        query,
        top_k=5,
        similarity_threshold=0.7 # Optional: Filter by similarity score
    )

    # 4. Use the results (e.g., provide context to an LLM)
    print(f"Found {len(results)} semantic results:")
    for i, result in enumerate(results):
        # result is a dict containing file_path, code snippet, score, etc.
        print(f"--- Result {i+1} ---")
        print(f"File: {result['file_path']}")
        print(f"Similarity: {result.get('similarity', 'N/A'):.4f}")
        print(f"Code Snippet:\n{result['code']}")
        print("-" * 20)

else:
    # Fall back to lexical search if semantic index isn't available
    print("Semantic search not available, falling back to lexical search.")
    results = codesleuth.search_lexically( # Use the main search method
        query, # Note: Lexical search uses the query as a pattern/regex
        max_results=5
    )
    print(f"Found {len(results)} lexical results:")
    for i, result in enumerate(results):
        # result is a dict containing file_path, line_number, matched_text, etc.
        print(f"--- Result {i+1} ---")
        print(f"File: {result['file_path']}")
        print(f"Line: {result['line_number']}")
        print(f"Matched Text:\n{result['matched_text']}")
        print("-" * 20)

```

## Configuration

CodeSleuth aims for smart defaults but allows fine-tuning via the `CodeSleuthConfig` object.

```python
from codesleuth.config import (
    CodeSleuthConfig,
    ParserConfig,
    IndexConfig,
    SearchConfig,
    EmbeddingModel,
)

# Example configuration (adjust values based on your needs and hardware)
config = CodeSleuthConfig(
    repo_path="/path/to/your/repo",

    # --- Parsing Configuration ---
    parser=ParserConfig(
        chunk_size=150,       # Lines per code chunk (affects context size)
        chunk_overlap=30,     # Lines overlapping between chunks (context continuity)
        ignore_patterns=[     # Regex patterns for files/dirs to ignore
            "node_modules/*",
            "dist/*",
            "build/*",
            "*.log",
            ".*\/", # Ignore hidden directories like .git, .vscode etc.
        ]
    ),

    # --- Indexing Configuration ---
    index=IndexConfig(
        model_name=EmbeddingModel.BGE_M3, # Embedding model (BGE_M3 supports MLX)
        # dimension=1024, # Dimension is usually inferred from the model
        use_mlx=True,         # Auto-use MLX on Apple Silicon if available
        use_gpu=False,        # Set True to attempt GPU use (PyTorch backend)
        batch_size=64,        # Embedding computation batch size (memory dependent)
        # FAISS HNSW parameters (used if not on Apple Silicon)
        hnsw_m=32,            # Connections per node (higher recall, slower build)
        hnsw_ef_construction=200, # Build-time search depth (higher quality, slower build)
        hnsw_ef_search=50     # Search-time depth (higher recall, slower search)
        # index_path=Path("./custom_codesleuth_index") # Optional: Custom index location
    ),

    # --- Search Configuration ---
    search=SearchConfig(
        max_results=10,         # Default max results for semantic search
        min_similarity=0.5,     # Default minimum similarity for semantic results
        max_grep_results=50     # Default max results for lexical (grep) search
    )
)

# Initialize CodeSleuth with the custom config
codesleuth = CodeSleuth(config)

# Now proceed with indexing and searching
# codesleuth.index_repository()
# results = codesleuth.search_semantically("your query")
```

**Key Configuration Parameters:**

-   `repo_path`: **Required.** Path to the code repository you want to index.
-   `parser.chunk_size`/`overlap`: Controls how code is split for embedding. Larger chunks capture more context but increase processing time.
-   `parser.ignore_patterns`: Crucial for excluding dependencies, build artifacts, logs, etc., to keep the index relevant and efficient.
-   `index.model_name`: Choose the embedding model. `BGE_M3` is recommended for its performance and MLX compatibility.
-   `index.use_mlx`: Set `True` to leverage Apple Silicon's Neural Engine (if available).
-   `index.batch_size`: Adjust based on your system's RAM/VRAM during indexing.
-   `index.hnsw_*`: Fine-tune the HNSW index performance (only applicable if _not_ on Apple Silicon).
-   `search.*`: Set default limits and thresholds for search results.

## Architecture

CodeSleuth performs code search through a multi-stage pipeline optimized for local execution:

1.  **Code Parsing**:
    -   Uses **Tree-sitter** for fast, syntax-aware parsing of supported languages.
    -   Intelligently chunks code, often at function/class boundaries, to preserve semantic context.
    -   Falls back to line-based chunking for plain text or unsupported languages.
    -   Applies configured `ignore_patterns` to skip irrelevant files/directories.
2.  **Embedding Generation**:
    -   Converts code chunks into dense vector embeddings using a chosen transformer model (e.g., `BGE_M3`).
    -   **MLX Optimization**: On Apple Silicon (M1/M2/M3), if `use_mlx=True` and `mlx-embedding-models` is installed, calculations are significantly accelerated using the Neural Engine.
    -   **PyTorch Backend**: On other platforms or if MLX is disabled/unavailable, uses Hugging Face Transformers via PyTorch (can utilize GPU if `use_gpu=True` and CUDA is available).
    -   Embeddings are processed in batches for efficiency.
3.  **Indexing**:
    -   Stores embeddings in a **FAISS** vector index for efficient similarity searches.
    -   **Platform-Aware Index Selection**:
        -   _Default (x86/AMD)_: Uses `faiss.IndexHNSWFlat`, an HNSW index optimized for speed and recall. Tunable via `hnsw_*` config parameters.
        -   _Apple Silicon (ARM)_: Uses `faiss.IndexFlatL2` for maximum compatibility and stability on M-series chips.
    -   `faiss.IndexIDMap` wraps the core index to link vectors back to their original code chunks.
    -   The index and metadata are saved locally (default: `./codesleuth_index`).
4.  **Search Execution**:
    -   **Semantic Search**:
        -   The natural language query is embedded using the same model.
        -   FAISS searches the index for vectors (code chunks) with the highest cosine similarity to the query vector.
        -   Results are returned ranked by similarity.
    -   **Lexical Search**:
        -   Delegates to the external `ripgrep` (rg) command-line tool.
        -   Executes `rg` with the provided pattern/regex against the repository path.
        -   Parses `rg`'s output to provide structured results (file path, line number, matched text).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See `CONTRIBUTING.md` for more details (if available).

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
