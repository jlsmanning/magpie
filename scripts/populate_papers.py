"""
Populate paper database from ArXiv.

Script to fetch papers from ArXiv and index them in ChromaDB.
Configurable via command-line arguments or config file.
"""

import argparse
import datetime
import typing

from magpie.data.arxiv_puller import fetch_papers, fetch_recent_papers
from magpie.data.paper_indexer import PaperIndexer
from magpie.integrations.embedder import Embedder


def populate_from_categories(
    categories: typing.List[str],
    start_date: datetime.date,
    end_date: datetime.date,
    max_results: int = 1000,
    batch_size: int = 100
) -> None:
    """
    Populate database with papers from specified categories and date range.
    
    Args:
        categories: ArXiv category codes (e.g., ["cs.AI", "cs.CV"])
        start_date: Earliest publication date
        end_date: Latest publication date
        max_results: Maximum total papers to fetch
        batch_size: Batch size for fetching from ArXiv
    """
    print(f"Populating database with papers from {start_date} to {end_date}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Max results: {max_results}\n")
    
    # Initialize indexer (creates embedder automatically)
    print("Initializing indexer and embedder...")
    indexer = PaperIndexer()
    print(f"Using embedder model: {indexer.embedder.get_model_name()}")
    print(f"Database path: {indexer.db_path}")
    print(f"Current paper count: {indexer.get_paper_count()}\n")
    
    # Fetch papers from ArXiv
    print("Fetching papers from ArXiv...")
    print("(This may take a while due to ArXiv API rate limiting)")
    
    papers = fetch_papers(
        categories=categories,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results
    )
    
    print(f"Fetched {len(papers)} papers from ArXiv\n")
    
    if not papers:
        print("No papers found. Exiting.")
        return
    
    # Index papers
    print("Indexing papers (generating embeddings and storing in database)...")
    stats = indexer.index_papers(papers, skip_existing=True)
    
    print(f"\nIndexing complete!")
    print(f"  Total papers: {stats['total']}")
    print(f"  Successfully indexed: {stats['indexed']}")
    print(f"  Skipped (already exist): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"\nFinal paper count in database: {indexer.get_paper_count()}")


def populate_recent(
    categories: typing.List[str],
    days_back: int = 7,
    max_results: int = 100
) -> None:
    """
    Populate database with recent papers from last N days.
    
    Convenience function for updating database with new papers.
    
    Args:
        categories: ArXiv category codes
        days_back: Number of days to look back
        max_results: Maximum papers to fetch
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    populate_from_categories(
        categories=categories,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results
    )


def main():
    """Command-line interface for populating paper database."""
    parser = argparse.ArgumentParser(
        description="Populate Magpie paper database from ArXiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate with AI/ML papers from last 2 years
  python scripts/populate_papers.py --categories cs.AI cs.LG cs.CV --years 2
  
  # Populate with recent papers from last week
  python scripts/populate_papers.py --categories cs.AI cs.LG --recent 7
  
  # Populate with specific date range
  python scripts/populate_papers.py --categories cs.AI --start 2023-01-01 --end 2024-01-01
        """
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="ArXiv categories to fetch (e.g., cs.AI cs.CV cs.LG)"
    )
    
    # Date range options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--years",
        type=int,
        help="Number of years back from today (e.g., --years 2 for last 2 years)"
    )
    date_group.add_argument(
        "--recent",
        type=int,
        help="Number of days back from today (e.g., --recent 7 for last week)"
    )
    date_group.add_argument(
        "--start",
        type=str,
        help="Start date in YYYY-MM-DD format (requires --end)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date in YYYY-MM-DD format (used with --start)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum number of papers to fetch (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Validate date arguments
    if args.start and not args.end:
        parser.error("--start requires --end")
    
    # Determine date range
    if args.years:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=args.years * 365)
    elif args.recent:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=args.recent)
    else:  # args.start and args.end
        try:
            start_date = datetime.datetime.strptime(args.start, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(args.end, "%Y-%m-%d").date()
        except ValueError as e:
            parser.error(f"Invalid date format: {e}")
    
    # Run population
    try:
        populate_from_categories(
            categories=args.categories,
            start_date=start_date,
            end_date=end_date,
            max_results=args.max_results
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
