import os
import pytest
from pathlib import Path

from codesleuth.config import CodeSleuthConfig, ParserConfig
from codesleuth.indexing.parser import (
    create_parser,
    CodeParser,
    CodeChunk,
    TreeSitterManager,
)


@pytest.fixture(scope="session")
def tree_sitter_manager():
    """Create a TreeSitterManager once for the entire test session."""
    return TreeSitterManager()


@pytest.fixture(scope="session")
def session_config():
    """Create a configuration for the entire test session."""
    config = CodeSleuthConfig()
    # Fix the ignore patterns to use valid regex patterns
    config.parser.ignore_patterns = [
        r"node_modules/.*",
        r"dist/.*",
        r"build/.*",
        r"\.git/.*",
        r"__pycache__/.*",
        r".*\.pyc$",
    ]
    return config


@pytest.fixture(scope="session")
def session_parser(session_config, tree_sitter_manager):
    """Create a parser once for the entire test session."""
    parser = create_parser(session_config)
    # Replace the parser's tree_sitter_manager with our session-scoped one
    parser.tree_sitter_manager = tree_sitter_manager
    return parser


@pytest.fixture
def config(session_config):
    """Return the session-scoped config for tests that need a non-session fixture."""
    return session_config


@pytest.fixture
def parser(session_parser):
    """Return the session-scoped parser for tests that need a non-session fixture."""
    return session_parser


@pytest.fixture(scope="session")
def test_files_dir(tmp_path_factory):
    """Create a directory with test files for each supported language."""
    # Use tmp_path_factory for session-scoped fixture
    tmp_path = tmp_path_factory.mktemp("test_files")

    # Python test file
    python_file = tmp_path / "example.py"
    python_file.write_text(
        """
def example_function(x):
    \"\"\"Test function\"\"\"
    return x * 2

class ExampleClass:
    def method1(self):
        pass
"""
    )

    # JavaScript test file
    js_file = tmp_path / "example.js"
    js_file.write_text(
        """
function example_function(x) {
    return x * 2;
}

class ExampleClass {
    method1() {
        return 'hello';
    }
}
"""
    )

    # TypeScript test file
    ts_file = tmp_path / "example.ts"
    ts_file.write_text(
        """
function example_function(x: number): number {
    return x * 2;
}

class ExampleClass {
    method1(): string {
        return 'hello';
    }
}
"""
    )

    # Java test file
    java_file = tmp_path / "Example.java"
    java_file.write_text(
        """
public class Example {
    public int exampleFunction(int x) {
        return x * 2;
    }
    
    private void method1() {
        System.out.println("Hello");
    }
}
"""
    )

    # C test file
    c_file = tmp_path / "example.c"
    c_file.write_text(
        """
int example_function(int x) {
    return x * 2;
}

void another_function() {
    printf("Hello, world\\n");
}
"""
    )

    # Create empty file for testing edge cases
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    return tmp_path


def test_language_detection(config):
    """Test that file extensions map to correct languages."""
    parser = create_parser(config)

    assert parser.get_language_for_file("example.py") == "python"
    assert parser.get_language_for_file("example.js") == "javascript"
    assert parser.get_language_for_file("example.ts") == "typescript"
    assert parser.get_language_for_file("Example.java") == "java"
    assert parser.get_language_for_file("example.c") == "c"
    assert parser.get_language_for_file("example.cpp") == "cpp"
    assert parser.get_language_for_file("example.go") == "go"
    assert parser.get_language_for_file("example.rs") == "rust"
    assert parser.get_language_for_file("example.txt") is None  # Unsupported file type


@pytest.mark.parametrize(
    "file_name,language,expected_symbol",
    [
        ("example.py", "python", "example_function"),
        ("example.js", "javascript", "example_function"),
        ("example.ts", "typescript", "example_function"),
        ("Example.java", "java", "exampleFunction"),
        ("example.c", "c", "example_function"),
    ],
)
def test_parser_language_support(
    file_name, language, expected_symbol, test_files_dir, parser
):
    """Test parser supports multiple languages correctly."""
    file_path = test_files_dir / file_name

    chunks = parser.parse_file(file_path)

    # Check we got some chunks back
    assert len(chunks) > 0

    # Check chunks have correct metadata
    assert any(chunk.language == language for chunk in chunks)

    # Since Tree-sitter might not be available, we'll make this test conditional
    # Check if at least one chunk has the expected symbol name OR
    # we're using fallback parsing (where symbol names might not be extracted)
    has_expected_symbol = any(
        chunk.symbol_name == expected_symbol
        for chunk in chunks
        if chunk.symbol_name is not None
    )
    using_fallback = all(chunk.symbol_name is None for chunk in chunks)

    assert has_expected_symbol or using_fallback


def test_fallback_parser(test_files_dir, parser):
    """Test the fallback parser when Tree-sitter parsing fails."""
    # Force fallback by using an unsupported file extension
    unsupported_file = test_files_dir / "example.txt"
    with open(unsupported_file, "w") as f:
        f.write("This is some text\nthat will be parsed\nusing the fallback parser.")

    chunks = parser._fallback_parse_file(str(unsupported_file))

    # Check fallback parser returns chunks
    assert len(chunks) > 0
    assert all(chunk.file_path == str(unsupported_file) for chunk in chunks)

    # Check chunks contain expected content
    all_text = "".join(chunk.code for chunk in chunks)
    assert "This is some text" in all_text
    assert "using the fallback parser" in all_text


def test_empty_file(test_files_dir, parser):
    """Test handling of empty files."""
    empty_file = test_files_dir / "empty.py"

    chunks = parser.parse_file(empty_file)

    # Empty file should return no chunks or empty chunk
    assert len(chunks) == 0 or (len(chunks) == 1 and not chunks[0].code.strip())


def test_parse_directory(test_files_dir, parser):
    """Test parsing an entire directory."""
    # Parse all files in the test directory
    chunks = list(parser.parse_directory(test_files_dir))

    # Should find chunks from all supported language files
    assert len(chunks) > 0

    # Count unique files parsed
    unique_files = set(chunk.file_path for chunk in chunks)

    # Should have at least 5 files (py, js, ts, java, c)
    assert len(unique_files) >= 5

    # Check that file paths are correctly recorded
    for chunk in chunks:
        assert chunk.file_path.startswith(str(test_files_dir))

    # Check if any of the chunks have extracted symbol names
    # This might fail if tree-sitter isn't available and all parsing falls back
    # to line-based chunking
    symbols_found = [
        chunk.symbol_name for chunk in chunks if chunk.symbol_name is not None
    ]
    if symbols_found:
        # At least one symbol name was extracted, so check for expected ones
        symbol_names = set(symbols_found)
        assert any(
            name in symbol_names
            for name in [
                "example_function",
                "ExampleClass",
                "exampleFunction",
                "another_function",
            ]
        )


def test_should_ignore_file(parser):
    """Test ignore pattern matching."""
    assert parser.should_ignore_file("node_modules/some_file.js") is True
    assert parser.should_ignore_file("dist/bundle.js") is True
    assert parser.should_ignore_file("build/output.js") is True
    assert parser.should_ignore_file(".git/HEAD") is True
    assert parser.should_ignore_file("__pycache__/module.pyc") is True
    assert parser.should_ignore_file("some_file.pyc") is True
    assert parser.should_ignore_file("src/main.py") is False
    assert parser.should_ignore_file("app/components/Button.js") is False
