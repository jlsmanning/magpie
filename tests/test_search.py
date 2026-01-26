"""
Test script for vector search functionality.

Demonstrates searching the paper database with sample queries.
"""

import datetime
from magpie.models.query import Query, SubQuery
from magpie.core.query_processor import process_query
from magpie.core.vector_search import VectorSearch
from magpie.integrations.embedder import Embedder


def test_single_query():
    """Test search with a single query."""
    print("=" * 80)
    print("TEST 1: Single Query Search")
    print("=" * 80)
    
    # Create a simple query
    query = Query(
        queries=[
            SubQuery(
                text="machine learning artificial intelligence",
                weight=1.0,
                source_interest_ids=None
            )
        ],
        max_results=5
    )
    
    print(f"\nSearching for: '{query.queries[0].text}'")
    print(f"Max results: {query.max_results}\n")
    
    # Process query (generate embeddings)
    print("Generating embeddings...")
    embedder = Embedder()
    processed_query = process_query(query, embedder)
    
    # Search database
    print("Searching database...")
    vector_search = VectorSearch()
    print(f"Database contains {vector_search.get_paper_count()} papers\n")
    
    results = vector_search.search(processed_query)
    
    # Display results
    print(f"Found {len(results)} results (from {results.total_found} total matches):\n")
    
    for i, result in enumerate(results.results, 1):
        paper = result.paper
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        if len(paper.authors) > 3:
            print(f"            (+ {len(paper.authors) - 3} more)")
        print(f"   Published: {paper.published_date}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        print(f"   URL: {paper.url}")
        print()


def test_multi_query():
    """Test search with multiple queries (different topics)."""
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Query Search with Deduplication")
    print("=" * 80)
    
    # Create multi-topic query
    query = Query(
        queries=[
            SubQuery(
                text="neural networks deep learning",
                weight=0.6,
                source_interest_ids=["interest-1"]
            ),
            SubQuery(
                text="computer vision image recognition",
                weight=0.4,
                source_interest_ids=["interest-2"]
            )
        ],
        max_results=10
    )
    
    print(f"\nSearching for:")
    for sq in query.queries:
        print(f"  - '{sq.text}' (weight: {sq.weight})")
    print(f"Max results: {query.max_results}\n")
    
    # Process and search
    print("Generating embeddings...")
    embedder = Embedder()
    processed_query = process_query(query, embedder)
    
    print("Searching database...")
    vector_search = VectorSearch()
    results = vector_search.search(processed_query)
    
    # Display results
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results.results, 1):
        paper = result.paper
        print(f"{i}. {paper.title}")
        print(f"   Relevance: {result.relevance_score:.3f}")
        print(f"   Matched queries: {len(result.matched_subqueries)}")
        for sq in result.matched_subqueries:
            print(f"     - '{sq.text}'")
        if result.matched_multiple_queries():
            print(f"   ⭐ BOOSTED (matched multiple queries)")
        print()


def test_with_filters():
    """Test search with date and citation filters (when implemented)."""
    print("\n" + "=" * 80)
    print("TEST 3: Search with Filters (TODO)")
    print("=" * 80)
    
    # This will work once filtering is implemented
    query = Query(
        queries=[
            SubQuery(
                text="artificial intelligence",
                weight=1.0,
                source_interest_ids=None
            )
        ],
        max_results=5,
        date_range=(datetime.date(2024, 1, 1), datetime.date(2025, 1, 1)),
        min_citations=10
    )
    
    print("\nQuery with filters:")
    print(f"  Text: '{query.queries[0].text}'")
    print(f"  Date range: {query.date_range}")
    print(f"  Min citations: {query.min_citations}")
    print("\nNote: Filtering not yet implemented in VectorSearch")


def main():
    """Run all tests."""
    try:
        test_single_query()
        test_multi_query()
        test_with_filters()
        
        print("\n" + "=" * 80)
        print("All tests complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise


if __name__ == "__main__":
    main()
