"""
Search results data models for Magpie.

Defines the structure of search results that flow through the pipeline.
Each PaperResult tracks which subqueries matched the paper for traceability.
"""

import datetime
import pydantic
import typing

from magpie.models.paper import Paper
from magpie.models.query import Query, SubQuery


class PaperResult(pydantic.BaseModel):
    """
    A single paper result with relevance information.
    
    Tracks which subqueries matched this paper, allowing tracing back to
    the user interests that generated those subqueries.
    """
    paper: Paper = pydantic.Field(
        ...,
        description="The paper that was found"
    )
    
    matched_subqueries: typing.List[SubQuery] = pydantic.Field(
        ...,
        description="Subqueries that returned this paper (can be multiple if paper matched multiple queries)",
        min_length=1
    )
    
    relevance_score: float = pydantic.Field(
        ...,
        description="Relevance score from vector search (possibly boosted if matched multiple subqueries)",
        ge=0.0,
        le=1.0
    )
    
    explanation: typing.Optional[str] = pydantic.Field(
        default=None,
        description="LLM-generated explanation of why this paper is relevant to user's interests"
    )
    
    def get_source_interest_ids(self) -> typing.Set[str]:
        """
        Get all interest IDs that led to this paper being found.
        Traces through all matched subqueries to their source interests.
        """
        interest_ids = set()
        for subquery in self.matched_subqueries:
            if subquery.source_interest_ids:
                interest_ids.update(subquery.source_interest_ids)
        return interest_ids
    
    def matched_multiple_queries(self) -> bool:
        """Check if this paper was found by multiple subqueries."""
        return len(self.matched_subqueries) > 1
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class SearchResults(pydantic.BaseModel):
    """
    Complete search results from the pipeline.
    
    Contains deduplicated papers with tracking of which subqueries matched each.
    Papers that appear in multiple subquery results have all matching subqueries listed.
    """
    results: typing.List[PaperResult] = pydantic.Field(
        default_factory=list,
        description="List of paper results, deduplicated by paper_id"
    )
    
    query: Query = pydantic.Field(
        ...,
        description="The query that generated these results"
    )
    
    total_found: int = pydantic.Field(
        ...,
        description="Total papers found before limiting to max_results",
        ge=0
    )
    
    timestamp: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now,
        description="When this search was performed"
    )
    
    def get_papers(self) -> typing.List[Paper]:
        """Extract just the Paper objects from results."""
        return [result.paper for result in self.results]
    
    def get_paper_ids(self) -> typing.Set[typing.Tuple[str, str]]:
        """Get set of all paper IDs in results."""
        return {result.paper.paper_id for result in self.results}
    
    def get_results_by_interest(self, interest_id: str) -> typing.List[PaperResult]:
        """
        Get all results that match a specific interest.
        Useful for grouping results by interest in the UI.
        """
        return [
            result for result in self.results
            if interest_id in result.get_source_interest_ids()
        ]
    
    def get_multi_query_matches(self) -> typing.List[PaperResult]:
        """Get papers that matched multiple subqueries (high-value results)."""
        return [result for result in self.results if result.matched_multiple_queries()]
    
    def __len__(self) -> int:
        """Number of results."""
        return len(self.results)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "paper": {
                            "paper_id": ["arxiv", "2301.12345"],
                            "title": "Grad-CAM: Visual Explanations from Deep Networks",
                            "authors": ["Jane Smith", "John Doe"],
                            "abstract": "We propose a technique for visual explanations...",
                            "published_date": "2024-01-15",
                            "url": "https://arxiv.org/abs/2301.12345",
                            "pdf_url": "https://arxiv.org/pdf/2301.12345.pdf",
                            "categories": ["cs.CV", "cs.AI"],
                            "citation_count": 150,
                            "venue": "CVPR 2024",
                            "metadata": {}
                        },
                        "matched_subqueries": [
                            {
                                "text": "computer vision explainability",
                                "weight": 0.6,
                                "source_interest_ids": ["interest-1", "interest-2"]
                            }
                        ],
                        "relevance_score": 0.89,
                        "explanation": "This paper directly addresses explainability in computer vision using visual attention mechanisms."
                    }
                ],
                "query": {
                    "queries": [
                        {
                            "text": "computer vision explainability",
                            "weight": 0.6,
                            "source_interest_ids": ["interest-1", "interest-2"]
                        }
                    ],
                    "max_results": 10,
                    "recency_weight": 0.7,
                    "exclude_seen_papers": True,
                    "original_input": "Find papers on XAI in computer vision"
                },
                "total_found": 47,
                "timestamp": "2025-01-21T10:30:00"
            }
        }
