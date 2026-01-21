"""
Paper data models for Magpie.

Defines paper metadata structure with fields common across paper sources.
Site-specific metadata is stored in the metadata dict.
"""

import datetime
import pydantic
import typing


class Paper(pydantic.BaseModel):
    """
    Research paper metadata from various sources (ArXiv, Semantic Scholar, etc.).
    
    Core fields are standardized across sources. Site-specific data goes in metadata.
    The source is encoded in paper_id as (source, id) tuple.
    """
    paper_id: typing.Tuple[str, str] = pydantic.Field(
        ...,
        description="Paper identifier as (source, id) tuple, e.g., ('arxiv', '2301.12345')"
    )
    
    title: str = pydantic.Field(
        ...,
        description="Paper title",
        min_length=1
    )
    
    authors: typing.List[str] = pydantic.Field(
        ...,
        description="typing.List of author names",
        min_length=1
    )
    
    abstract: str = pydantic.Field(
        ...,
        description="Paper abstract",
        min_length=1
    )
    
    published_date: datetime.date = pydantic.Field(
        ...,
        description="Date paper was published"
    )
    
    url: str = pydantic.Field(
        ...,
        description="Canonical URL to paper page"
    )
    
    pdf_url: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Direct URL to PDF download"
    )
    
    doi: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Digital Object Identifier (DOI)"
    )
    
    categories: typing.Optional[typing.List[str]] = pydantic.Field(
        default=None,
        description="Subject categories/topics (e.g., ArXiv categories like 'cs.CV')"
    )
    
    citation_count: typing.Optional[int] = pydantic.Field(
        default=None,
        description="Number of citations (if available)",
        ge=0
    )
    
    venue: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Publication venue (conference/journal name)"
    )
    
    metadata: typing.Dict[str, typing.Any] = pydantic.Field(
        default_factory=dict,
        description="Site-specific metadata that doesn't fit in standard fields"
    )
    
    @pydantic.field_validator('title', 'abstract')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure text fields are not just whitespace."""
        if not v.strip():
            raise ValueError("Text field cannot be empty or whitespace")
        return v.strip()
    
    @property
    def source(self) -> str:
        """Get the source (first element of paper_id tuple)."""
        return self.paper_id[0]
    
    @property
    def id(self) -> str:
        """Get the ID (second element of paper_id tuple)."""
        return self.paper_id[1]
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{self.title} ({authors_str}, {self.published_date.year})"
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "paper_id": ["arxiv", "2301.12345"],
                "title": "Explaining Deep Neural Networks with Grad-CAM",
                "authors": ["Jane Smith", "John Doe", "Alice Johnson"],
                "abstract": "We propose a technique for producing visual explanations...",
                "published_date": "2024-01-15",
                "url": "https://arxiv.org/abs/2301.12345",
                "pdf_url": "https://arxiv.org/pdf/2301.12345.pdf",
                "doi": "10.48550/arXiv.2301.12345",
                "categories": ["cs.CV", "cs.AI", "cs.LG"],
                "citation_count": 42,
                "venue": "CVPR 2024",
                "metadata": {
                    "arxiv_comment": "Accepted at CVPR 2024. Code available.",
                    "arxiv_primary_category": "cs.CV",
                    "updated_date": "2024-03-10"
                }
            }
        }
