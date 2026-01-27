"""
Result synthesis for Magpie.

Generates explanations for why papers are relevant and organizes results.
"""

import typing

from magpie.models.results import SearchResults, PaperResult
from magpie.models.profile import UserProfile


def synthesize_results(
    results: SearchResults,
    profile: UserProfile,
    llm_client=None  # Stub for future LLM-based synthesis
) -> SearchResults:
    """
    Generate explanations for search results.
    
    Populates the explanation field for each PaperResult based on:
    - Which user interests led to this paper
    - Which subqueries matched
    - Paper metadata (categories, venue, recency)
    
    TODO: Implement LLM-based synthesis that:
    - Generates natural language explanations of relevance
    - Summarizes key contributions from abstract
    - Explains connections to user's specific interests
    
    Args:
        results: SearchResults from reranker
        profile: User profile with interests
        llm_client: LLM client for generating explanations (not yet implemented)
        
    Returns:
        SearchResults with explanation fields populated
    """
    if not results.results:
        return results
    
    # Generate explanation for each paper
    for paper_result in results.results:
        paper_result.explanation = _generate_basic_explanation(
            paper_result,
            profile
        )
        
        # TODO: Replace basic explanation with LLM-generated one
        # llm_explanation = _generate_llm_explanation(
        #     paper_result,
        #     profile,
        #     results.query,
        #     llm_client
        # )
        # paper_result.explanation = llm_explanation
    
    return results


def _generate_basic_explanation(
    paper_result: PaperResult,
    profile: UserProfile
) -> str:
    """
    Generate basic explanation from metadata (no LLM).
    
    Args:
        paper_result: Paper with matched subqueries
        profile: User profile for interest lookup
        
    Returns:
        Explanation string
    """
    paper = paper_result.paper
    
    # Build explanation parts
    parts = []
    
    # Which queries matched
    if paper_result.matched_multiple_queries():
        query_texts = [sq.text for sq in paper_result.matched_subqueries]
        parts.append(f"Matches multiple interests: {', '.join(query_texts)}")
    else:
        query_text = paper_result.matched_subqueries[0].text
        parts.append(f"Matches: {query_text}")
    
    # Which user interests (if traceable)
    interest_ids = paper_result.get_source_interest_ids()
    if interest_ids:
        interest_names = []
        for interest_id in interest_ids:
            interest = profile.get_interest_by_id(interest_id)
            if interest:
                interest_names.append(interest.topic)
        if interest_names:
            parts.append(f"Related to your interests in: {', '.join(interest_names)}")
    
    # Venue if notable
    if paper.venue:
        parts.append(f"Published in {paper.venue}")
    
    # Categories
    if paper.categories:
        primary_cat = paper.categories[0]
        parts.append(f"Category: {primary_cat}")
    
    # Citation count if significant
    if paper.citation_count and paper.citation_count >= 50:
        parts.append(f"Highly cited ({paper.citation_count} citations)")
    
    # Recency
    age_days = (datetime.date.today() - paper.published_date).days
    if age_days < 90:
        parts.append("Recent publication")
    
    return ". ".join(parts) + "."


def _generate_llm_explanation(
    paper_result: PaperResult,
    profile: UserProfile,
    query,
    llm_client
) -> str:
    """
    Generate rich explanation using LLM (stub).
    
    TODO: Implement LLM call that:
    1. Provides paper title, abstract, metadata
    2. Provides user interests and query context
    3. Asks LLM to explain:
       - Why this paper is relevant
       - How it relates to user's interests
       - Key contributions/findings
       - Whether user should prioritize reading it
    
    Args:
        paper_result: Paper with relevance info
        profile: User profile with interests
        query: Original query
        llm_client: LLM client
        
    Returns:
        Natural language explanation
    """
    # Stub: Would make LLM call here
    raise NotImplementedError("LLM-based synthesis not yet implemented")


# Import datetime for age calculation
import datetime
