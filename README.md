# CodeSleuth

CodeSleuth is a local-first code search and retrieval tool optimized for LLM integration. It creates semantic embeddings for code snippets and provides fast, efficient search capabilities without requiring external services or cloud dependencies.

## Key Features

### 🚀 Local-First Architecture

-   **Zero External Dependencies**: All processing happens locally on your machine
-   **No Cloud Services**: Your code never leaves your system
-   **Fast Indexing**: Optimized for large codebases
-   **Efficient Storage**: Compact index format for quick loading

### 🧠 LLM-Optimized

-   **Semantic Search**: Find code based on natural language descriptions
-   **Context-Aware Results**: Perfect for LLM code generation and analysis
-   **Natural Language Queries**: Search using everyday language
-   **Code Understanding**: Built for LLM code comprehension tasks

### ⚡ Performance Optimizations

-   **MLX Integration**: Native support for Apple Silicon
-   **Smart Index Selection**:
    -   FAISS with HNSW for x86/AMD architectures
    -   Optimized FAISS for ARM processors
-   **Efficient Embedding**: Fast code chunk processing
-   **Memory-Efficient**: Smart chunking and caching

### 🔍 Search Capabilities

-   **Semantic Search**: Find code by meaning, not just text
-   **Lexical Search**: Precise text-based search with regex support
-   **Function Definition Search**: Quickly locate function and method definitions
-   **Reference Search**: Find all usages of a symbol
-   **File Search**: Search for files by name or pattern

### 🌐 Language Support

Supports a wide range of programming languages including:

-   Python
-   JavaScript/TypeScript
-   Java
-   C/C++
-   PHP
-   Go
-   Rust
-   And more...

## Installation

```bash
pip install codesleuth
```

## Quick Start

```python
from codesleuth import CodeSleuth
from codesleuth.config import CodeSleuthConfig

# Initialize CodeSleuth with your repository path
config = CodeSleuthConfig(repo_path="/path/to/your/repo")
codesleuth = CodeSleuth(config)

# Index your repository (uses MLX on Apple Silicon, FAISS otherwise)
codesleuth.index_repository()

# Search for code semantically (perfect for LLM integration)
results = codesleuth.search_semantically(
    "authentication service implementation",
    top_k=5,
    similarity_threshold=0.7
)

# Use with your LLM
for result in results:
    print(f"Found relevant code in {result['file_path']}:")
    print(result['code'])
```

## Configuration

CodeSleuth automatically optimizes for your hardware:

```python
from codesleuth.config import CodeSleuthConfig, ParserConfig, IndexConfig

config = CodeSleuthConfig(
    repo_path="/path/to/repo",
    parser=ParserConfig(
        chunk_size=100,
        chunk_overlap=20,
        ignore_patterns=["node_modules/*", "dist/*"]
    ),
    # Index configuration is automatically optimized for your hardware
    index=IndexConfig(
        embedding_model="bge-m3",  # Uses MLX on Apple Silicon
        dimension=1024
    )
)
```

## Advanced Usage

### LLM Integration

```python
# Example with an LLM
from codesleuth import CodeSleuth
from your_llm import LLM

codesleuth = CodeSleuth(config)
llm = LLM()

# Search for relevant code
results = codesleuth.search_semantically(
    "implement user authentication with JWT",
    top_k=3
)

# Use the results with your LLM
context = "\n".join(result["code"] for result in results)
response = llm.generate(f"Based on this code:\n{context}\n\nImplement a similar authentication system.")
```

### Performance Tuning

```python
# Optimize for your specific use case
config = CodeSleuthConfig(
    repo_path="/path/to/repo",
    parser=ParserConfig(
        chunk_size=150,  # Larger chunks for better semantic understanding
        chunk_overlap=30,  # More overlap for better context
    ),
    index=IndexConfig(
        embedding_model="bge-m3",
        dimension=1024,
        hnsw_ef_construction=200,  # Tune HNSW index quality
        hnsw_ef_search=50  # Tune search speed vs accuracy
    )
)
```

## Architecture

CodeSleuth is built with performance in mind:

1. **Code Parsing**: Uses Tree-sitter for fast, accurate code parsing
2. **Embedding Generation**:
    - MLX on Apple Silicon for native performance
    - Optimized FAISS on other architectures
3. **Index Storage**: Efficient binary format for quick loading
4. **Search**: HNSW-based similarity search for fast results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
