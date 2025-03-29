"""Command-line interface for CodeSleuth."""

import typer
from pathlib import Path
from typing import Optional

from codesleuth import CodeSleuth
from codesleuth.config import CodeSleuthConfig

app = typer.Typer(help="CodeSleuth - Intelligent Code Search and Analysis")


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to the repository to index"),
    chunk_size: int = typer.Option(100, help="Size of code chunks for indexing"),
    chunk_overlap: int = typer.Option(20, help="Overlap between chunks"),
):
    """Index a repository for code search."""
    config = CodeSleuthConfig(
        repo_path=str(repo_path),
        parser=ParserConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )
    codesleuth = CodeSleuth(config)
    codesleuth.index_repository()
    typer.echo(f"Successfully indexed repository at {repo_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    repo_path: Path = typer.Argument(..., help="Path to the indexed repository"),
    top_k: int = typer.Option(5, help="Number of results to return"),
    similarity_threshold: float = typer.Option(0.7, help="Minimum similarity score"),
    search_type: str = typer.Option(
        "semantic", help="Type of search (semantic/lexical)"
    ),
):
    """Search for code in the repository."""
    config = CodeSleuthConfig(repo_path=str(repo_path))
    codesleuth = CodeSleuth(config)

    if search_type == "semantic":
        results = codesleuth.search_semantically(
            query, top_k=top_k, similarity_threshold=similarity_threshold
        )
    else:
        results = codesleuth.search_lexically(query, max_results=top_k)

    for result in results:
        typer.echo(f"\nFile: {result['file_path']}")
        typer.echo(f"Lines: {result['start_line']}-{result['end_line']}")
        if result.get("symbol_name"):
            typer.echo(f"Symbol: {result['symbol_name']}")
        typer.echo("Code:")
        typer.echo(result["code"])
        typer.echo(f"Similarity: {result['similarity']:.2f}")


@app.command()
def references(
    symbol: str = typer.Argument(..., help="Symbol to find references for"),
    repo_path: Path = typer.Argument(..., help="Path to the indexed repository"),
    definition: bool = typer.Option(
        False, help="Search for definitions instead of references"
    ),
):
    """Find references to a symbol in the repository."""
    config = CodeSleuthConfig(repo_path=str(repo_path))
    codesleuth = CodeSleuth(config)
    results = codesleuth.search_references(symbol, definition=definition)

    for result in results:
        typer.echo(f"\nFile: {result['file_path']}")
        typer.echo(f"Lines: {result['start_line']}-{result['end_line']}")
        typer.echo("Code:")
        typer.echo(result["code"])


@app.command()
def metadata(
    file_path: Path = typer.Argument(..., help="Path to the file"),
    repo_path: Path = typer.Argument(..., help="Path to the repository"),
):
    """Get metadata about a file."""
    config = CodeSleuthConfig(repo_path=str(repo_path))
    codesleuth = CodeSleuth(config)
    metadata = codesleuth.get_code_metadata(str(file_path))

    typer.echo(f"\nMetadata for {file_path}:")
    typer.echo("\nFunctions:")
    for func in metadata["functions"]:
        typer.echo(f"  - {func}")
    typer.echo("\nClasses:")
    for cls in metadata["classes"]:
        typer.echo(f"  - {cls}")


@app.command()
def view(
    file_path: Path = typer.Argument(..., help="Path to the file"),
    repo_path: Path = typer.Argument(..., help="Path to the repository"),
    start_line: Optional[int] = typer.Option(None, help="Starting line number"),
    end_line: Optional[int] = typer.Option(None, help="Ending line number"),
):
    """View file contents."""
    config = CodeSleuthConfig(repo_path=str(repo_path))
    codesleuth = CodeSleuth(config)
    content = codesleuth.view_file(str(file_path), start_line, end_line)
    typer.echo(content)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
