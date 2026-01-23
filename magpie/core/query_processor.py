"""
Query processing for Magpie.

Converts Query objects into embeddings for vector search.
Handles query expansion and embedding generation.

TODO: Add support for boolean query operators (AND/OR/NOT)
      for more precise retrieval when semantic search isn't enough.
"""

import typing
import dataclasses
import numpy
from magpie.models.query import Query
from magpie.integrations.embedder import Embedder


@dataclasses.dataclass
class ProcessedQuery:
    """
    Query with embeddings computed for each subquery.
    
    IMPORTANT: embeddings list corresponds exactly to query.queries by index.
    That is, embeddings[i] is the embedding vector for query.queries[i].
    
    This index-based correspondence must be maintained throughout the pipeline.
    The vector search component relies on this ordering to match results back
    to their source subqueries.
    """
    query: Query
    embeddings: typing.List[numpy.ndarray]  # embeddings[i] ↔ query.queries[i]
    
    def __post_init__(self):
        """Validate that embeddings match number of subqueries."""
        if len(self.embeddings) != len(self.query.queries):
            raise ValueError(
                f"Number of embeddings ({len(self.embeddings)}) must match "
                f"number of subqueries ({len(self.query.queries)})"
            )


def process_query(query: Query, embedder: Embedder) -> ProcessedQuery:
    """
    Convert Query into embeddings for vector search.
    
    Embeds each subquery text using the provided embedder. Query expansion
    with synonyms/related terms is currently stubbed (TODO).
    
    Args:
        query: Query object containing subqueries to embed
        embedder: Embedder instance for generating embeddings
        
    Returns:
        ProcessedQuery where embeddings[i] corresponds to query.queries[i]
        by index. This correspondence MUST be preserved when passing to
        vector search.
        
    Example:
        >>> embedder = Embedder()
        >>> query = Query(queries=[
        ...     SubQuery(text="explainable AI", weight=0.6),
        ...     SubQuery(text="computer vision", weight=0.4)
        ... ])
        >>> processed = process_query(query, embedder)
        >>> # processed.embeddings[0] is embedding for "explainable AI"
        >>> # processed.embeddings[1] is embedding for "computer vision"
    """
    texts = [subquery.text for subquery in query.queries]
    
    embeddings = embedder.embed_batch(texts)
    
    # Convert to list of individual arrays for clearer indexing
    embeddings_list = [embeddings[i] for i in range(len(embeddings))]
    
    return ProcessedQuery(
        query=query,
        embeddings=embeddings_list
    )


def expand_query_text(text: str) -> str:
    """
    Expand query text with synonyms and related terms.
    
    TODO: Implement query expansion using:
    - Synonym lookup (WordNet or similar)
    - Related terms from domain knowledge
    - LLM-based expansion for better semantic coverage
    
    For now, returns text unchanged.
    
    Args:
        text: Original query text
        
    Returns:
        Expanded query text with additional terms
    """
    # Stub: No expansion yet
    return text
