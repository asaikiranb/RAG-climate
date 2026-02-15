#!/usr/bin/env python3
"""
CLI search tool for querying the document collection.

Usage:
    python scripts/search.py "What is the India Cooling Action Plan?"
    python scripts/search.py "low-GWP refrigerants" --top-k 10
    python scripts/search.py "passive cooling" --export results.json
"""

import argparse
import sys
from pathlib import Path
import json
import time
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.retrieval import HybridRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_result(idx: int, result: Dict, show_full: bool = False):
    """
    Pretty print a single search result.

    Args:
        idx: Result number
        result: Result dictionary
        show_full: Whether to show full document text
    """
    metadata = result.get('metadata', {})
    score = metadata.get('score', result.get('score', 0.0))

    print(f"\n[{idx}] Score: {score:.4f}")

    # Show metadata
    print(f"    File:  {metadata.get('filename', 'Unknown')}")
    print(f"    Page:  {metadata.get('page_number', '?')}")

    # Show retrieval metadata if available
    if 'retrieval_metadata' in result:
        ret_meta = result['retrieval_metadata']
        print(f"    Retrieved with: {ret_meta.get('query_variations', 1)} query variations")
        if ret_meta.get('reranked'):
            print(f"    Reranked: Yes")
        if ret_meta.get('diversified'):
            print(f"    Diversified: Yes")

    # Show document preview or full text
    document = result.get('document', result.get('text', ''))
    if show_full:
        print(f"\n{document}")
    else:
        preview = document[:300] + "..." if len(document) > 300 else document
        print(f"\n{preview}")

    print("-" * 80)


def format_results_table(results: List[Dict]):
    """Format results as a simple table."""
    print("\n" + "-" * 80)
    print(f"{'#':<4} {'Score':<8} {'File':<30} {'Page':<6} {'Preview':<30}")
    print("-" * 80)

    for idx, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        score = metadata.get('score', result.get('score', 0.0))
        filename = metadata.get('filename', 'Unknown')[:28]
        page = str(metadata.get('page_number', '?'))
        document = result.get('document', result.get('text', ''))
        preview = document[:28].replace('\n', ' ')

        print(f"{idx:<4} {score:<8.4f} {filename:<30} {page:<6} {preview:<30}")

    print("-" * 80)


def export_results(results: List[Dict], output_path: Path, pretty: bool = True):
    """
    Export results to JSON file.

    Args:
        results: Search results
        output_path: Output file path
        pretty: Whether to pretty-print JSON
    """
    try:
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)

        print(f"\n✓ Results exported to: {output_path}")
        print(f"  Total results: {len(results)}")

    except Exception as e:
        print(f"\nERROR: Failed to export results: {e}")
        logger.error(f"Export failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Search the document collection using hybrid retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python scripts/search.py "What is the Montreal Protocol?"

  # Return more results
  python scripts/search.py "low-GWP refrigerants" --top-k 10

  # Show full document text
  python scripts/search.py "passive cooling strategies" --full

  # Disable advanced features for faster search
  python scripts/search.py "HVAC maintenance" --simple

  # Export results to JSON
  python scripts/search.py "India Cooling Action Plan" --export results.json

  # Table format
  python scripts/search.py "refrigerant alternatives" --table
        """
    )

    parser.add_argument(
        'query',
        type=str,
        help='Search query'
    )

    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Show full document text instead of preview'
    )

    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple search (disable HyDE, reranking, etc.)'
    )

    parser.add_argument(
        '--table',
        action='store_true',
        help='Display results as a table'
    )

    parser.add_argument(
        '--export',
        type=Path,
        metavar='FILE',
        help='Export results to JSON file'
    )

    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Include detailed retrieval metadata in output'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Initialize
    print_header(f"HYBRID SEARCH: {args.query}")

    try:
        settings = get_settings()

        # Disable advanced features if --simple flag is set
        if args.simple:
            settings.retrieval.use_hyde = False
            settings.retrieval.use_query_expansion = False
            settings.retrieval.use_reranking = False
            settings.retrieval.use_mmr = False
            print("Running simple search (advanced features disabled)")

        print(f"Collection: {settings.chromadb.collection_name}")
        if not args.simple:
            print(f"Features:   HyDE={settings.retrieval.use_hyde}, "
                  f"Expansion={settings.retrieval.use_query_expansion}, "
                  f"Rerank={settings.retrieval.use_reranking}, "
                  f"MMR={settings.retrieval.use_mmr}")

    except Exception as e:
        print(f"ERROR: Failed to initialize settings: {e}")
        return 1

    # Initialize retriever
    try:
        print("Initializing retriever...")
        retriever = HybridRetriever(settings)
    except Exception as e:
        print(f"ERROR: Failed to initialize retriever: {e}")
        logger.error(f"Retriever initialization failed: {e}")
        return 1

    # Perform search
    try:
        start_time = time.time()

        results = retriever.search(
            query=args.query,
            top_k=args.top_k,
            return_metadata=args.metadata
        )

        elapsed = time.time() - start_time

        if not results:
            print("\nNo results found.")
            return 0

        print(f"\nFound {len(results)} results in {elapsed:.2f}s")

        # Display results
        if args.table:
            format_results_table(results)
        else:
            for idx, result in enumerate(results, 1):
                print_result(idx, result, show_full=args.full)

        # Export if requested
        if args.export:
            export_results(results, args.export, pretty=True)

        # Show search stats
        print(f"\nSearch completed in {elapsed:.2f}s")

        return 0

    except Exception as e:
        print(f"\nERROR: Search failed: {e}")
        logger.error(f"Search failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
