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
from tree_sitter import Tree
from tree_sitter_language_pack import get_language, get_parser

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
    SUPPORTED_LANGUAGES = [
        "python",
        "javascript",
        "typescript",
        "php",
        "cpp",
        "c",
        "java",
        "go",
        "rust",
    ]

    def __init__(self):
        """Initialize the TreeSitterManager."""
        self.languages = {}
        self.parsers = {}
        self._setup_languages()

    def _setup_languages(self):
        """Set up Tree-sitter languages from tree-sitter-language-pack."""
        for lang_name in self.SUPPORTED_LANGUAGES:
            try:
                # Get language and parser from tree-sitter-language-pack
                language = get_language(lang_name)
                parser = get_parser(lang_name)

                # Store them in our dictionaries
                self.languages[lang_name] = language
                self.parsers[lang_name] = parser
                logger.info(f"Loaded parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to load language {lang_name}: {e}")

    def get_parser(self, language: str) -> Optional[object]:
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
            List of code chunks
        """
        str_path = str(file_path)
        logger.debug(f"Parsing file: {str_path}")

        # Get the language from the file extension
        language = self.get_language_for_file(str_path)
        if language is None:
            # For text files, use a generic text language identifier
            if str_path.endswith(
                (
                    ".txt",
                    ".md",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".toml",
                    ".ini",
                    ".cfg",
                    ".conf",
                )
            ):
                language = "text"
                logger.debug(f"Treating {str_path} as text file")
            else:
                logger.warning(f"Unknown language for file: {str_path}")
                return []

        logger.debug(f"Detected language: {language}")

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
                logger.debug(f"File contains {line_count} lines")

            # For text files, always use line-based chunking
            if language == "text":
                return self._fallback_parse_file(str_path, language)

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
                    logger.debug(f"Found {len(chunks)} chunks using Tree-sitter")
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
        logger.debug(f"Parsing {file_path} with Tree-sitter")
        tree = self.tree_sitter_manager.parse_code(code, language)
        if tree is None:
            logger.warning(f"Failed to parse {file_path} with Tree-sitter")
            return []

        chunks = []
        root_node = tree.root_node
        logger.debug(f"Tree root node type: {root_node.type}")

        # Define query patterns for different languages to find function-like nodes
        query_patterns = {
            "python": """
                (function_definition
                  name: (identifier) @function_name) @function
                  
                (class_definition
                  name: (identifier) @class_name
                  body: (block
                    (function_definition
                      name: (identifier) @method_name) @method)) @class
                  
                (class_definition
                  name: (identifier) @class_name
                  body: (block
                    (function_definition
                      name: (identifier) @method_name
                      body: (block
                        (expression_statement
                          (call
                            function: (attribute
                              object: (identifier) @self
                              attribute: (identifier) @method_call)))))) @class_method
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
                logger.debug(f"Found {len(captures)} captures in {file_path}")

                # Group captures by their node to find function/method definitions
                nodes_with_names = {}
                # The structure of captures might be different in tree-sitter-language-pack
                # Handle both formats: (node, name) tuples or objects with node and name attributes
                for capture in captures:
                    try:
                        if isinstance(capture, tuple) and len(capture) == 2:
                            # Original format: (node, name)
                            node, capture_name = capture
                        else:
                            # New format might be an object with attributes
                            node = (
                                capture.node if hasattr(capture, "node") else capture[0]
                            )
                            capture_name = (
                                capture.name if hasattr(capture, "name") else capture[1]
                            )

                        node_id = id(node)
                        if node_id not in nodes_with_names:
                            nodes_with_names[node_id] = {"node": node, "captures": {}}

                        if capture_name.endswith("_name"):
                            nodes_with_names[node_id]["captures"][capture_name] = (
                                node.text.decode("utf-8")
                            )
                        elif capture_name in [
                            "function",
                            "method",
                            "class",
                            "arrow_function",
                        ]:
                            nodes_with_names[node_id]["type"] = capture_name
                    except Exception as e:
                        logger.error(
                            f"Error processing query capture: {e} - Capture: {capture}"
                        )
                        continue

                logger.debug(
                    f"Found {len(nodes_with_names)} unique nodes in {file_path}"
                )

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
                        logger.debug(
                            f"Created chunk for {name} ({node_type}) at lines {start_line}-{end_line}"
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
                (
                    r"\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*->",
                    "method",
                ),  # Class methods with return type
                (
                    r"\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:",
                    "method",
                ),  # Class methods without return type
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
                    logger.debug(
                        f"Created chunk for {symbol_name} ({symbol_type}) at lines {start_line}-{end_line}"
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
