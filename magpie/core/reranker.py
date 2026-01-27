"""
Result reranking for Magpie.

Reranks search results based on multiple factors beyond embedding similarity.
Combines vector similarity with recency, citations, venue quality, and LLM evaluation.
"""

import datetime
import typing

from magpie.models.results import SearchResults, PaperResult


def rerank_results(
    results: SearchResults,
    llm_client=None  # Stub for future LLM-based semantic evaluation
) -> SearchResults:
    """
    Rerank search results using multiple ranking factors.
    
    Applies non-LLM ranking adjustments based on:
    - Paper recency (query.recency_weight)
    - Citation count (query.min_citations)
    - Venue quality (query.venues)
    
    TODO: Add LLM-based semantic evaluation that considers:
    - Deep relevance beyond embedding similarity
    - Paper quality indicators from abstract
    - Alignment with user's specific research interests
    
    Args:
        results: SearchResults from vector search
        llm_client: LLM client for semantic evaluation (not yet implemented)
        
    Returns:
        SearchResults with adjusted relevance scores and reordered papers
    """
    if not results.results:
        return results
    
    query = results.query
    
    # Adjust scores for each paper
    for paper_result in results.results:
        paper = paper_result.paper
        base_score = paper_result.relevance_score
        
        # Calculate adjustment factors
        recency_bonus = _calculate_recency_bonus(
            paper.published_date,
            query.recency_weight
        )
        
        citation_bonus = _calculate_citation_bonus(
            paper.citation_count,
            query.min_citations
        )
        
        venue_bonus = _calculate_venue_bonus(
            paper.venue,
            query.venues
        )
        
        # TODO: Reconsider this scoring formula
        # Current approach: additive adjustments to base relevance score
        # Alternatives to explore:
        # - Multiplicative factors (score * recency_mult * citation_mult)
        # - Weighted combination with learnable weights
        # - Non-linear transformations
        adjusted_score = base_score + recency_bonus + citation_bonus + venue_bonus
        
        # Clamp to [0, 1] range
        paper_result.relevance_score = max(0.0, min(1.0, adjusted_score))
    
    # Re-sort by adjusted scores
    results.results.sort(key=lambda pr: pr.relevance_score, reverse=True)
    
    # TODO: Add LLM-based evaluation here
    # This would:
    # 1. Format paper + query for LLM
    # 2. Ask LLM to evaluate relevance and quality
    # 3. Generate explanation (populate paper_result.explanation)
    # 4. Final score adjustment based on LLM assessment
    
    return results


def _calculate_recency_bonus(
    published_date: datetime.date,
    recency_weight: float
) -> float:
    """
    Calculate bonus for paper recency.
    
    More recent papers get higher bonus, scaled by recency_weight.
    
    Args:
        published_date: When paper was published
        recency_weight: Weight for recency factor (0-1)
        
    Returns:
        Recency bonus to add to relevance score
    """
    if recency_weight == 0.0:
        return 0.0
    
    # Calculate age in days
    today = datetime.date.today()
    age_days = (today - published_date).days
    
    # Papers from last year get full bonus, older papers get decreasing bonus
    # Normalize by 2 years (730 days)
    age_factor = max(0.0, 1.0 - (age_days / 730.0))
    
    # Scale by recency_weight and cap bonus at 0.2
    bonus = age_factor * recency_weight * 0.2
    
    return bonus


def _calculate_citation_bonus(
    citation_count: typing.Optional[int],
    min_citations: typing.Optional[int]
) -> float:
    """
    Calculate bonus for citation count.
    
    Highly-cited papers get bonus. Papers below min_citations threshold
    get penalty.
    
    Args:
        citation_count: Number of citations (None for unknown)
        min_citations: Minimum citation threshold from query
        
    Returns:
        Citation bonus/penalty to add to relevance score
    """
    if citation_count is None:
        # No citation data available - no adjustment
        return 0.0
    
    # Penalty for papers below minimum threshold
    if min_citations is not None and citation_count < min_citations:
        return -0.1
    
    # Bonus for highly-cited papers (logarithmic scale)
    # 10 citations: +0.05, 100 citations: +0.10, 1000 citations: +0.15
    if citation_count > 0:
        import math
        bonus = min(0.15, math.log10(citation_count) * 0.05)
        return bonus
    
    return 0.0


def _calculate_venue_bonus(
    venue: typing.Optional[str],
    preferred_venues: typing.Optional[typing.List[str]]
) -> float:
    """
    Calculate bonus for publication venue.
    
    Papers from preferred venues get bonus.
    
    Args:
        venue: Publication venue name (e.g., "CVPR 2024")
        preferred_venues: List of preferred venue names
        
    Returns:
        Venue bonus to add to relevance score
    """
    if not venue or not preferred_venues:
        return 0.0
    
    venue_lower = venue.lower()
    
    # Check if paper venue matches any preferred venue
    for preferred in preferred_venues:
        if preferred.lower() in venue_lower:
            return 0.1  # Fixed bonus for preferred venue
    
    return 0.0
