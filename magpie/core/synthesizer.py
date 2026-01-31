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
    llm_client=None
) -> SearchResults:
    """
    Generate explanations for search results.
    
    Uses LLM to create 2-3 sentence explanations for each paper,
    describing relevance and key contributions. Optimized for audio
    presentation - brief enough to listen to while deciding to explore further.
    
    Args:
        results: SearchResults from reranker
        profile: User profile with research context
        llm_client: LLM client for generating explanations (creates if None)
        
    Returns:
        SearchResults with explanation fields populated
    """
    if not results.results:
        return results
    
    # Use LLM to generate explanations
    if llm_client is None:
        from magpie.integrations.llm_client import ClaudeClient
        llm_client = ClaudeClient()
    
    # Generate explanations in batch
    try:
        explanations = _generate_llm_explanations(
            results=results,
            profile=profile,
            llm_client=llm_client
        )
        
        # Apply explanations to paper results
        for i, paper_result in enumerate(results.results):
            if i < len(explanations):
                paper_result.explanation = explanations[i]
            else:
                # Fallback to basic explanation if LLM didn't return enough
                paper_result.explanation = _generate_basic_explanation(
                    paper_result,
                    profile
                )
    
    except Exception as e:
        # If LLM synthesis fails, fall back to basic explanations
        print(f"Warning: LLM synthesis failed: {e}")
        for paper_result in results.results:
            paper_result.explanation = _generate_basic_explanation(
                paper_result,
                profile
            )
    
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


def _generate_llm_explanations(
    results: SearchResults,
    profile: UserProfile,
    llm_client
) -> typing.List[str]:
    """
    Generate explanations for all papers using LLM in batch.
    
    Args:
        results: SearchResults with papers to explain
        profile: User profile with research context
        llm_client: LLM client
        
    Returns:
        List of explanation strings (same order as results.results)
    """
    # Build system prompt
    system_prompt = _build_synthesis_prompt(results.query, profile)
    
    # Format papers for synthesis
    papers_text = _format_papers_for_synthesis(results.results)
    
    user_message = f"""
Generate brief explanations for these papers. For each paper, write 2-3 sentences that:
1. Explain why it's relevant to the user's query
2. Describe the key contribution or approach

Keep it concise - user will listen to these to decide which papers to explore further.

Papers:
{papers_text}

Return JSON:
{{
  "explanations": [
    "Paper 0: explanation here...",
    "Paper 1: explanation here...",
    ...
  ]
}}
"""
    
    response = llm_client.chat_with_json(
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
        max_tokens=3000,
        temperature=0.3  # Slightly creative but consistent
    )
    
    return response.get("explanations", [])


def _build_synthesis_prompt(
    query: 'Query',
    profile: UserProfile
) -> str:
    """
    Build system prompt for synthesis.
    
    Args:
        query: Search query
        profile: User profile with context
        
    Returns:
        System prompt string
    """
    query_topics = [sq.text for sq in query.queries]
    query_text = ", ".join(query_topics)
    
    user_context = profile.research_context or "No specific context provided."
    
    prompt = f"""You are explaining research papers to a user.

USER QUERY: {query_text}

USER RESEARCH CONTEXT:
{user_context}

Your task is to generate brief, audio-friendly explanations (2-3 sentences each).

Each explanation should:
- State why the paper is relevant to the query
- Describe the key contribution or approach
- Be conversational and easy to understand when spoken aloud
- Help user decide if they want to explore this paper further

The user will listen to these while walking/driving, so be concise and clear.
"""
    
    return prompt


def _format_papers_for_synthesis(
    papers: typing.List['PaperResult']
) -> str:
    """
    Format papers for synthesis.
    
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
            lines.append(f"  (+ {len(paper.authors) - 3} more)")
        lines.append(f"Published: {paper.published_date}")
        if paper.venue:
            lines.append(f"Venue: {paper.venue}")
        
        # Include which queries matched
        matched_queries = [sq.text for sq in paper_result.matched_subqueries]
        lines.append(f"Matched queries: {', '.join(matched_queries)}")
        
        # Include reranker's reasoning if available
        if paper_result.rerank_reasoning:
            lines.append(f"Reranker said: {paper_result.rerank_reasoning}")
        
        lines.append(f"Abstract: {paper.abstract}")
        lines.append("")
    
    return "\n".join(lines)



# Import datetime for age calculation
import datetime

# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from magpie.models.results import PaperResult
    from magpie.models.query import Query
