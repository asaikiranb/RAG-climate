#!/usr/bin/env python3
"""
CLI script for document ingestion.

Usage:
    python scripts/ingest.py /path/to/documents/
    python scripts/ingest.py /path/to/document.pdf
    python scripts/ingest.py --reset  # Clear and re-ingest
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.ingestion import IngestionPipeline
from src.core.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_stats(stats: dict):
    """Pretty print ingestion statistics."""
    print_header("INGESTION SUMMARY")

    if 'total_files' in stats:
        # Folder stats
        print(f"  Total Files:       {stats['total_files']}")
        print(f"  Successful:        {stats['successful']}")
        print(f"  Failed:            {stats['failed']}")
        print(f"  Skipped:           {stats['skipped']}")
        print(f"  Total Pages:       {stats['total_pages']}")
        print(f"  Total Chunks:      {stats['total_chunks']}")

        if stats['total_chunks'] > 0:
            avg_chunks = stats['total_chunks'] / max(stats['successful'], 1)
            print(f"  Avg Chunks/File:   {avg_chunks:.1f}")
    else:
        # Single file stats
        print(f"  Status:            {stats['status']}")
        print(f"  Filename:          {stats.get('filename', 'N/A')}")
        print(f"  Pages:             {stats.get('pages', 0)}")
        print(f"  Chunks:            {stats.get('chunks', 0)}")
        print(f"  Method:            {stats.get('method', 'N/A')}")

    print("=" * 70 + "\n")


def reset_collection(settings):
    """Reset (delete and recreate) the vector store collection."""
    print_header("RESETTING COLLECTION")

    try:
        vector_store = VectorStore(settings)
        collection_name = settings.chromadb.collection_name

        # Delete collection
        print(f"Deleting collection: {collection_name}...")
        vector_store.client.delete_collection(name=collection_name)
        print("Collection deleted successfully.")

        # Recreate
        print(f"Recreating collection: {collection_name}...")
        vector_store = VectorStore(settings)
        print("Collection recreated successfully.")

        print("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        print(f"ERROR: {e}")
        return False


def show_collection_stats(settings):
    """Display current collection statistics."""
    try:
        pipeline = IngestionPipeline(settings)
        stats = pipeline.get_collection_stats()

        print_header("CURRENT COLLECTION STATS")
        print(f"  Collection:        {stats['collection_name']}")
        print(f"  Total Documents:   {stats['total_documents']}")
        print(f"  Embedding Model:   {settings.embedding.model_name}")
        print(f"  Chunking Method:   {settings.chunking.method}")
        print(f"  Chunk Size:        {settings.chunking.chunk_size} tokens")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        print(f"ERROR: Could not retrieve collection stats: {e}")


def ingest_path(path: Path, pipeline: IngestionPipeline, verbose: bool = False):
    """
    Ingest a file or folder.

    Args:
        path: Path to file or folder
        pipeline: IngestionPipeline instance
        verbose: Whether to show detailed progress

    Returns:
        Ingestion statistics dictionary
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    start_time = time.time()

    if path.is_file():
        # Ingest single file
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"Only PDF files are supported. Got: {path.suffix}")

        print(f"\nIngesting file: {path.name}")
        stats = pipeline.ingest_pdf(path)

    elif path.is_dir():
        # Ingest folder
        print(f"\nIngesting folder: {path}")
        stats = pipeline.ingest_folder(path)

    else:
        raise ValueError(f"Invalid path type: {path}")

    elapsed = time.time() - start_time
    stats['elapsed_time'] = elapsed

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all PDFs from a folder
  python scripts/ingest.py data/pdfs/

  # Ingest a single PDF
  python scripts/ingest.py data/pdfs/document.pdf

  # Reset collection and re-ingest
  python scripts/ingest.py data/pdfs/ --reset

  # Show current collection stats
  python scripts/ingest.py --stats

  # Verbose output
  python scripts/ingest.py data/pdfs/ -v
        """
    )

    parser.add_argument(
        'path',
        type=Path,
        nargs='?',
        help='Path to PDF file or folder containing PDFs'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset (delete and recreate) the collection before ingestion'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show current collection statistics and exit'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Initialize
    print_header("DOCUMENT INGESTION")

    try:
        settings = get_settings()
        print(f"Collection: {settings.chromadb.collection_name}")
        print(f"Embedding:  {settings.embedding.model_name}")
        print(f"Chunking:   {settings.chunking.method} ({settings.chunking.chunk_size} tokens)")
        print()

    except Exception as e:
        print(f"ERROR: Failed to initialize settings: {e}")
        return 1

    # Handle --stats flag
    if args.stats:
        show_collection_stats(settings)
        return 0

    # Require path unless showing stats
    if not args.path:
        parser.print_help()
        return 1

    # Handle --reset flag
    if args.reset:
        if not reset_collection(settings):
            return 1

    # Initialize pipeline
    try:
        pipeline = IngestionPipeline(settings)
    except Exception as e:
        print(f"ERROR: Failed to initialize ingestion pipeline: {e}")
        logger.error(f"Pipeline initialization failed: {e}")
        return 1

    # Ingest documents
    try:
        stats = ingest_path(args.path, pipeline, verbose=args.verbose)
        print_stats(stats)

        if 'elapsed_time' in stats:
            print(f"Total time: {stats['elapsed_time']:.1f}s\n")

        # Show final collection stats
        show_collection_stats(settings)

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    except Exception as e:
        print(f"ERROR: Ingestion failed: {e}")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
