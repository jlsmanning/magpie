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
    profile: 'UserProfile',
    llm_client=None,
    rerank_top_n: int = 20
) -> SearchResults:
    """
    Rerank search results using multiple ranking factors.
    
    Applies both non-LLM adjustments (recency, citations, venue) and
    LLM-based semantic evaluation for deep relevance assessment.
    
    Args:
        results: SearchResults from vector search
        profile: User profile with research context
        llm_client: LLM client for semantic evaluation (optional, creates if None)
        rerank_top_n: Number of top papers to evaluate with LLM (default: 20)
        
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
    
    # Apply LLM-based reranking to top N candidates
    if llm_client is not None and rerank_top_n > 0:
        results = _llm_rerank_top_papers(
            results=results,
            profile=profile,
            llm_client=llm_client,
            top_n=rerank_top_n
        )
    
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

#TODO: If citation count is None, retrieve it from somewhere?
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


def _llm_rerank_top_papers(
    results: SearchResults,
    profile: 'UserProfile',
    llm_client,
    top_n: int
) -> SearchResults:
    """
    Use LLM to re-evaluate top N papers for deep semantic relevance.
    
    Sends batch of papers to LLM with abstracts, vector similarity scores,
    and user context. LLM evaluates actual relevance and assigns new scores.
    
    Args:
        results: SearchResults with initial ranking
        profile: User profile with research context
        llm_client: LLM client for evaluation
        top_n: Number of top papers to rerank
        
    Returns:
        SearchResults with LLM-adjusted scores for top papers
    """
    from magpie.integrations.llm_client import ClaudeClient
    
    if llm_client is None:
        llm_client = ClaudeClient()
    
    # Only rerank top N papers
    papers_to_rerank = results.results[:top_n]
    
    if not papers_to_rerank:
        return results
    
    # Build prompt with papers
    system_prompt = _build_reranking_prompt(
        query=results.query,
        profile=profile
    )
    
    papers_text = _format_papers_for_reranking(papers_to_rerank)
    
    user_message = f"""
Evaluate these papers for relevance to the query. For each paper, provide:
1. A relevance score from 0.0 to 1.0
2. Brief reasoning (one sentence)

Papers to evaluate:
{papers_text}

Return JSON:
{{
  "evaluations": [
    {{"paper_index": 0, "score": 0.85, "reasoning": "..."}},
    ...
  ]
}}
"""
    
    try:
        response = llm_client.chat_with_json(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            max_tokens=2048,
            temperature=0.0  # Deterministic for consistent scoring
        )
        
        # Apply LLM scores to papers
        evaluations = response.get("evaluations", [])
        for eval_data in evaluations:
            idx = eval_data.get("paper_index")
            new_score = eval_data.get("score", 0.0)
            reasoning = eval_data.get("reasoning", "")
            
            if idx is not None and 0 <= idx < len(papers_to_rerank):
                papers_to_rerank[idx].relevance_score = new_score
                papers_to_rerank[idx].rerank_reasoning = reasoning
        
        # Re-sort all results (LLM-reranked papers + unchanged papers)
        results.results.sort(key=lambda pr: pr.relevance_score, reverse=True)
        
    except Exception as e:
        # If LLM reranking fails, continue with existing scores
        print(f"Warning: LLM reranking failed: {e}")
    
    return results


def _build_reranking_prompt(
    query: 'Query',
    profile: 'UserProfile'
) -> str:
    """
    Build system prompt for LLM reranking.
    
    Args:
        query: Search query with subqueries
        profile: User profile with research context
        
    Returns:
        System prompt string
    """
    # Extract query topics
    query_topics = [sq.text for sq in query.queries]
    query_text = ", ".join(query_topics)
    
    # User context
    user_context = profile.research_context or "No specific context provided."
    
    prompt = f"""You are evaluating research papers for relevance to a user's search query.

USER QUERY: {query_text}

USER RESEARCH CONTEXT:
{user_context}

EVALUATION CRITERIA:
1. Actual relevance - Does the paper truly address the query, or just use similar keywords?
2. Quality indicators - Venue reputation, methodological rigor, novelty vs. incremental work
3. User alignment - Does it match the user's research context and preferences?
4. Practical value - Does it provide actionable insights or just theoretical analysis?

You will receive papers with their vector similarity scores (based on embedding similarity to query).
Your job is to evaluate whether the similarity score accurately reflects TRUE relevance.

- If a paper has high similarity but low actual relevance, downgrade it
- If a paper has lower similarity but is exceptionally relevant/high-quality, upgrade it
- Consider the abstract content, not just keywords

Assign each paper a relevance score from 0.0 (not relevant) to 1.0 (highly relevant).
"""
    
    return prompt


def _format_papers_for_reranking(
    papers: typing.List['PaperResult']
) -> str:
    """
    Format papers for LLM evaluation.
    
    Args:
        papers: List of PaperResult objects
        
    Returns:
        Formatted string with paper details
    """
    lines = []
    for i, paper_result in enumerate(papers):
        paper = paper_result.paper
        
        lines.append(f"\n--- Paper {i} ---")
        lines.append(f"Title: {paper.title}")
        lines.append(f"Authors: {', '.join(paper.authors[:3])}")
        if len(paper.authors) > 3:
            lines.append(f"  (+ {len(paper.authors) - 3} more authors)")
        lines.append(f"Published: {paper.published_date}")
        if paper.venue:
            lines.append(f"Venue: {paper.venue}")
        if paper.citation_count:
            lines.append(f"Citations: {paper.citation_count}")
        lines.append(f"Vector similarity: {paper_result.relevance_score:.3f}")
        lines.append(f"Abstract: {paper.abstract}")
        lines.append("")
    
    return "\n".join(lines)


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from magpie.models.results import PaperResult
    from magpie.models.query import Query
    from magpie.models.profile import UserProfile
