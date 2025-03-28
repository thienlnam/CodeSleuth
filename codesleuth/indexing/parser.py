"""
Parser module for CodeSleuth.

This module contains functions to parse code files into chunks using Tree-sitter
or fallback methods when Tree-sitter is not available for a language.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from loguru import logger
from tree_sitter import Language, Parser, Tree

from ..config import CodeSleuthConfig, ParserConfig


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""

    file_path: str
    start_line: int
    end_line: int
    code: str
    symbol_name: Optional[str] = None
    language: Optional[str] = None


class TreeSitterManager:
    """Manages Tree-sitter language parsers."""

    # Map of supported languages to their grammar repositories
    LANGUAGE_REPOS = {
        "python": "tree-sitter-python",
        "javascript": "tree-sitter-javascript",
        "typescript": "tree-sitter-typescript",
        "php": "tree-sitter-php",
        "cpp": "tree-sitter-cpp",
        "c": "tree-sitter-c",
        "java": "tree-sitter-java",
        "go": "tree-sitter-go",
        "rust": "tree-sitter-rust",
    }

    REPO_URLS = {
        "tree-sitter-python": "https://github.com/tree-sitter/tree-sitter-python",
        "tree-sitter-javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
        "tree-sitter-typescript": "https://github.com/tree-sitter/tree-sitter-typescript",
        "tree-sitter-php": "https://github.com/tree-sitter/tree-sitter-php",
        "tree-sitter-cpp": "https://github.com/tree-sitter/tree-sitter-cpp",
        "tree-sitter-c": "https://github.com/tree-sitter/tree-sitter-c",
        "tree-sitter-java": "https://github.com/tree-sitter/tree-sitter-java",
        "tree-sitter-go": "https://github.com/tree-sitter/tree-sitter-go",
        "tree-sitter-rust": "https://github.com/tree-sitter/tree-sitter-rust",
    }

    def __init__(self):
        """Initialize the TreeSitterManager."""
        self.languages: Dict[str, Language] = {}
        self.parsers: Dict[str, Parser] = {}
        self.build_dir = Path(__file__).parent / "build"
        self.build_dir.mkdir(exist_ok=True, parents=True)
        self.language_so = self.build_dir / "languages.so"
        self._setup_languages()

    def _setup_languages(self):
        """Set up Tree-sitter languages."""
        # Check if we need to build language library
        needs_build = not self.language_so.exists()

        # If languages.so exists, try to load languages
        if not needs_build:
            try:
                self._load_languages_from_so()
            except Exception as e:
                logger.warning(f"Failed to load languages from shared library: {e}")
                needs_build = True

        # Build language library if needed
        if needs_build:
            logger.info("Building Tree-sitter language library...")
            self._build_languages()
            try:
                self._load_languages_from_so()
            except Exception as e:
                logger.error(f"Failed to load languages after building: {e}")

        # Create parsers for available languages
        for lang_name, lang in self.languages.items():
            parser = Parser()
            parser.set_language(lang)
            self.parsers[lang_name] = parser
            logger.info(f"Loaded parser for {lang_name}")

    def _load_languages_from_so(self):
        """Load languages from the shared library."""
        for lang_name in self.LANGUAGE_REPOS:
            try:
                lang = Language(self.language_so, lang_name)
                self.languages[lang_name] = lang
                logger.info(f"Loaded language {lang_name} from shared library")
            except Exception as e:
                logger.warning(f"Failed to load language {lang_name}: {e}")

    def _build_languages(self):
        """
        Build the languages shared library.

        Note: This is a simplified version that assumes git and build tools are available.
        In a real implementation, this would need more robust error handling and possibly
        platform-specific code.
        """
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Build each language
            successfully_built = []

            for lang_name, repo_name in self.LANGUAGE_REPOS.items():
                repo_url = self.REPO_URLS.get(repo_name)
                if not repo_url:
                    logger.warning(f"No repository URL found for {repo_name}")
                    continue

                try:
                    logger.info(f"Cloning {repo_name}...")
                    subprocess.run(
                        ["git", "clone", "--depth=1", repo_url, tmp_path / repo_name],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                    # Special case for TypeScript which has multiple parsers
                    if lang_name == "typescript":
                        lang_dir = tmp_path / repo_name / "typescript"
                    else:
                        lang_dir = tmp_path / repo_name

                    # Add this language to be built
                    successfully_built.append((lang_name, lang_dir))

                except Exception as e:
                    logger.error(f"Failed to clone {repo_name}: {e}")

            # Now build the languages
            if successfully_built:
                try:
                    from tree_sitter import build_library

                    # Build the library with the languages we successfully cloned
                    language_dirs = [
                        str(lang_dir) for _, lang_dir in successfully_built
                    ]
                    build_library(self.language_so, language_dirs)

                    logger.info(
                        f"Successfully built languages: {[lang for lang, _ in successfully_built]}"
                    )
                except Exception as e:
                    logger.error(f"Failed to build language library: {e}")

    def get_parser(self, language: str) -> Optional[Parser]:
        """
        Get a parser for the specified language.

        Args:
            language: Language name

        Returns:
            Parser for the language or None if not available
        """
        return self.parsers.get(language)

    def parse_code(self, code: str, language: str) -> Optional[Tree]:
        """
        Parse code using the appropriate parser.

        Args:
            code: Code to parse
            language: Language name

        Returns:
            Parse tree or None if parsing fails
        """
        parser = self.get_parser(language)
        if parser is None:
            return None

        try:
            tree = parser.parse(bytes(code, "utf8"))
            return tree
        except Exception as e:
            logger.error(f"Error parsing code with {language} parser: {e}")
            return None


class CodeParser:
    """Parser for code files."""

    def __init__(self, config: CodeSleuthConfig):
        """
        Initialize the CodeParser.

        Args:
            config: CodeSleuth configuration
        """
        self.config = config
        self.parser_config = config.parser
        self.tree_sitter_manager = TreeSitterManager()

    def get_language_for_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Determine the language for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None if unknown
        """
        ext = os.path.splitext(str(file_path))[1].lower()
        return self.parser_config.extension_to_language.get(ext)

    def should_ignore_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file should be ignored based on ignore patterns.

        Args:
            file_path: Path to the file

        Returns:
            True if the file should be ignored, False otherwise
        """
        str_path = str(file_path)
        for pattern in self.parser_config.ignore_patterns:
            if re.search(pattern, str_path):
                return True
        return False

    def parse_file(self, file_path: Union[str, Path]) -> List[CodeChunk]:
        """
        Parse a file into code chunks.

        Args:
            file_path: Path to the file

        Returns:
            List of CodeChunk objects
        """
        str_path = str(file_path)
        language = self.get_language_for_file(str_path)

        if language is None:
            logger.warning(f"Unknown language for file: {str_path}")
            return self._fallback_parse_file(str_path)

        parser = self.tree_sitter_manager.get_parser(language)
        if parser is None:
            logger.warning(
                f"No parser available for {language}, falling back to line-based chunking"
            )
            return self._fallback_parse_file(str_path, language)

        try:
            # Try to use Tree-sitter for function-level chunking
            with open(str_path, "r", encoding="utf-8") as f:
                code = f.read()
                line_count = code.count("\n") + 1

            # Parse the file with Tree-sitter
            # In an actual implementation, this would parse the tree and extract functions
            # For now, we'll simulate it with a placeholder that still falls back
            if language in ["python", "javascript", "typescript", "java"]:
                # These languages have well-defined function structures we could parse
                # This is a placeholder for the actual implementation
                chunks = self._parse_functions_tree_sitter(
                    str_path, language, code, line_count
                )
                if chunks:
                    return chunks

            # If function parsing fails or is not supported, fall back to line-based chunking
            logger.info(f"Falling back to line-based chunking for {str_path}")
            return self._fallback_parse_file(str_path, language)
        except Exception as e:
            logger.error(f"Error parsing file with Tree-sitter: {e}")
            return self._fallback_parse_file(str_path, language)

    def _parse_functions_tree_sitter(
        self, file_path: str, language: str, code: str, line_count: int
    ) -> List[CodeChunk]:
        """
        Parse functions using Tree-sitter.

        Args:
            file_path: Path to the file
            language: Language of the file
            code: File content
            line_count: Total line count

        Returns:
            List of CodeChunk objects
        """
        tree = self.tree_sitter_manager.parse_code(code, language)
        if tree is None:
            return []

        chunks = []
        root_node = tree.root_node

        # Define query patterns for different languages to find function-like nodes
        query_patterns = {
            "python": """
                (function_definition
                  name: (identifier) @function_name) @function
                  
                (class_definition
                  name: (identifier) @class_name) @class
            """,
            "javascript": """
                (function_declaration
                  name: (identifier) @function_name) @function
                
                (method_definition
                  name: (property_identifier) @method_name) @method
                  
                (arrow_function
                  parameter: (identifier) @param_name) @arrow_function
                  
                (class_declaration
                  name: (identifier) @class_name) @class
            """,
            "typescript": """
                (function_declaration
                  name: (identifier) @function_name) @function
                
                (method_definition
                  name: (property_identifier) @method_name) @method
                  
                (arrow_function
                  parameter: (identifier) @param_name) @arrow_function
                  
                (class_declaration
                  name: (identifier) @class_name) @class
            """,
            "java": """
                (method_declaration
                  name: (identifier) @method_name) @method
                  
                (class_declaration
                  name: (identifier) @class_name) @class
            """,
        }

        # If we have a query pattern for this language, use it
        if language in query_patterns:
            try:
                # Try to use Tree-sitter query to find functions/methods
                from tree_sitter import Query

                query = Query(
                    self.tree_sitter_manager.languages[language],
                    query_patterns[language],
                )
                captures = query.captures(root_node)

                # Group captures by their node to find function/method definitions
                nodes_with_names = {}
                for capture, capture_name in captures:
                    node_id = id(capture)
                    if node_id not in nodes_with_names:
                        nodes_with_names[node_id] = {"node": capture, "captures": {}}

                    if capture_name.endswith("_name"):
                        nodes_with_names[node_id]["captures"][capture_name] = (
                            capture.text.decode("utf-8")
                        )
                    elif capture_name in [
                        "function",
                        "method",
                        "class",
                        "arrow_function",
                    ]:
                        nodes_with_names[node_id]["type"] = capture_name

                # Create chunks for each function/method
                for node_data in nodes_with_names.values():
                    if "type" in node_data:
                        node = node_data["node"]
                        node_type = node_data["type"]

                        # Find an appropriate name
                        name = None
                        for capture_name, capture_value in node_data.get(
                            "captures", {}
                        ).items():
                            if capture_name.endswith("_name"):
                                name = capture_value
                                break

                        # If no name, use a placeholder
                        if name is None:
                            if node_type == "arrow_function":
                                name = "anonymous_arrow_function"
                            else:
                                name = f"anonymous_{node_type}"

                        # Calculate line range
                        start_point = node.start_point
                        end_point = node.end_point
                        start_line = start_point[0] + 1  # Convert to 1-indexed
                        end_line = end_point[0] + 1  # Convert to 1-indexed

                        # Extract the code
                        start_byte = node.start_byte
                        end_byte = node.end_byte
                        node_code = code[start_byte:end_byte]

                        chunks.append(
                            CodeChunk(
                                file_path=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                code=node_code,
                                symbol_name=name,
                                language=language,
                            )
                        )

                return chunks

            except Exception as e:
                logger.error(f"Error querying Tree-sitter tree: {e}")
                # Fall back to regex-based parsing

        # If Tree-sitter query fails or is not supported, fall back to regex-based parsing
        return self._regex_parse_functions(file_path, language, code, line_count)

    def _regex_parse_functions(
        self, file_path: str, language: str, code: str, line_count: int
    ) -> List[CodeChunk]:
        """
        Parse functions using regex patterns as a fallback method.

        Args:
            file_path: Path to the file
            language: Language of the file
            code: File content
            line_count: Total line count

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        if language == "python":
            # Python functions and classes
            patterns = [
                (r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
                (r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\(|\:)", "class"),
            ]

            for pattern, symbol_type in patterns:
                matches = list(re.finditer(pattern, code))

                for i, match in enumerate(matches):
                    symbol_name = match.group(1)
                    start_pos = match.start()

                    # Find the end of the function/class (start of next symbol or end of file)
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(code)

                    # Convert positions to line numbers
                    start_line = code[:start_pos].count("\n") + 1
                    end_line = code[:end_pos].count("\n") + 1

                    # Extract function/class code
                    symbol_code = code[start_pos:end_pos]

                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            code=symbol_code,
                            symbol_name=symbol_name,
                            language=language,
                        )
                    )

        elif language in ["javascript", "typescript"]:
            # JS/TS functions, methods, and classes
            patterns = [
                (r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
                (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function\s*\(", "function"),
                (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*=>", "arrow_function"),
                (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{", "method"),
                (r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", "class"),
            ]

            for pattern, symbol_type in patterns:
                matches = list(re.finditer(pattern, code))

                for i, match in enumerate(matches):
                    symbol_name = match.group(1)
                    start_pos = match.start()

                    # Find the end (using scope matching would be better but is complex for regex)
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(code)

                    # Convert positions to line numbers
                    start_line = code[:start_pos].count("\n") + 1
                    end_line = code[:end_pos].count("\n") + 1

                    # Extract code
                    symbol_code = code[start_pos:end_pos]

                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            code=symbol_code,
                            symbol_name=symbol_name,
                            language=language,
                        )
                    )

        elif language == "java":
            # Java methods and classes
            patterns = [
                (
                    r"(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])",
                    "method",
                ),
                (r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", "class"),
            ]

            for pattern, symbol_type in patterns:
                matches = list(re.finditer(pattern, code))

                for i, match in enumerate(matches):
                    if symbol_type == "method":
                        symbol_name = match.group(2)  # Method name is in group 2
                    else:
                        symbol_name = match.group(1)  # Class name is in group 1

                    start_pos = match.start()

                    # Find end
                    if i < len(matches) - 1:
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(code)

                    # Convert to line numbers
                    start_line = code[:start_pos].count("\n") + 1
                    end_line = code[:end_pos].count("\n") + 1

                    # Extract code
                    symbol_code = code[start_pos:end_pos]

                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            code=symbol_code,
                            symbol_name=symbol_name,
                            language=language,
                        )
                    )

        return chunks

    def _fallback_parse_file(
        self, file_path: str, language: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Fall back to line-based chunking when Tree-sitter parsing is not available.

        Args:
            file_path: Path to the file
            language: Language of the file, if known

        Returns:
            List of CodeChunk objects
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []

        chunks = []
        total_lines = len(lines)

        # Create chunks with overlap
        for i in range(
            0,
            total_lines,
            self.parser_config.chunk_size - self.parser_config.chunk_overlap,
        ):
            end_line = min(i + self.parser_config.chunk_size, total_lines)
            chunk_lines = lines[i:end_line]
            chunk_text = "".join(chunk_lines)

            # Only create chunks with actual content
            if chunk_text.strip():
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=i + 1,  # 1-indexed line numbers
                        end_line=end_line,
                        code=chunk_text,
                        language=language,
                    )
                )

            if end_line >= total_lines:
                break

        return chunks

    def parse_directory(self, directory: Union[str, Path]) -> Iterator[CodeChunk]:
        """
        Recursively parse all code files in a directory.

        Args:
            directory: Directory to parse

        Yields:
            CodeChunk objects for each chunk in each file
        """
        dir_path = Path(directory)

        for root, dirs, files in os.walk(dir_path):
            # Skip ignored directories
            dirs[:] = [
                d for d in dirs if not self.should_ignore_file(os.path.join(root, d))
            ]

            for file in files:
                file_path = os.path.join(root, file)

                if self.should_ignore_file(file_path):
                    continue

                # Skip files that are too large
                if os.path.getsize(file_path) > 1_000_000:  # 1MB
                    logger.warning(f"Skipping large file: {file_path}")
                    continue

                try:
                    chunks = self.parse_file(file_path)
                    for chunk in chunks:
                        yield chunk
                except Exception as e:
                    logger.error(f"Error parsing file {file_path}: {e}")


def create_parser(config: CodeSleuthConfig) -> CodeParser:
    """
    Create a CodeParser from a configuration.

    Args:
        config: CodeSleuth configuration

    Returns:
        CodeParser instance
    """
    return CodeParser(config)
