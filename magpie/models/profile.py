"""
User profile data models for Magpie.

Defines user research interests and default search preferences.
"""

import datetime
import pydantic
import typing
if typing.TYPE_CHECKING:
    from magpie.models.query import Query

class UserProfile(pydantic.BaseModel):
    """
    Complete user profile containing research interests and search preferences.
    
    Default search parameters can be overridden per-query.
    """
    user_id: str = pydantic.Field(
        ...,
        description="Unique identifier for this user"
    ) #FIXME: ensure unique
    
   
    # Default search parameters (can be overridden in Query)
    max_results: int = pydantic.Field(
        default=10,
        description="Default number of papers to return",
        ge=1,
        le=100
    )
    
    date_range: typing.Optional[typing.Tuple[datetime.date, datetime.date]] = pydantic.Field(
        default=None,
        description="Default date range filter for papers"
    )
    
    min_citations: typing.Optional[int] = pydantic.Field(
        default=None,
        description="Default minimum citation count",
        ge=0
    )
    
    recency_weight: float = pydantic.Field(
        default=0.5,
        description="Default weight for recent papers vs highly-cited (0-1)",
        ge=0.0,
        le=1.0
    )
    
    venues: typing.Optional[typing.Set[str]] = pydantic.Field(
        default=None,
        description="Default preferred publication venues"
    )
    
    exclude_seen_papers: bool = pydantic.Field(
        default=True,
        description="Default: filter out previously seen papers"
    )

    current_query: typing.Optional["Query"] = pydantic.Field(
        default=None,
        description="Current query being built/modified before execution"
    )

    research_context: typing.Optional[str] = pydantic.Field(
        default=None,
        description="Summary of user's research goals, expertise level, and preferences for paper discovery"
    )

    active_review_session: typing.Optional[typing.Dict[str, typing.Any]] = pydantic.Field(
        default=None,
        description="Serialized active review session for resume capability"
    )
    
    # History tracking
    seen_papers: typing.Dict[str, datetime.datetime] = pydantic.Field(
        default_factory=dict,
        description="Papers user has seen, mapped to when they were seen"
    )
    
    # Profile metadata
    created_at: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now,
        description="When this profile was created"
    )
    
    last_updated: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now,
        description="When this profile was last modified"
    )
    
    @pydantic.field_validator('date_range')
    @classmethod
    def validate_date_range(cls, v: typing.Optional[typing.Tuple[datetime.date, datetime.date]]) -> typing.Optional[typing.Tuple[datetime.date, datetime.date]]:
        """Ensure start date is before end date."""
        if v is not None:
            start_date, end_date = v
            if start_date > end_date:
                raise ValueError(f"Start date {start_date} must be before end date {end_date}")
        return v
    
    def mark_paper_seen(self, paper_id: str) -> None:
        """Mark a paper as seen."""
        self.seen_papers[paper_id] = datetime.datetime.now()
        self.last_updated = datetime.datetime.now()
    
    def has_seen_paper(self, paper_id: str) -> bool:
        """Check if user has already seen a paper."""
        return paper_id in self.seen_papers

    def cleanup_old_seen_papers(self, days: int = 90) -> None: #FIXME: ensure I'm actually using this!
        """Remove papers seen more than N days ago."""
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        self.seen_papers = {
            pid: when for pid, when in self.seen_papers.items()
            if when > cutoff
        }
        self.last_updated = datetime.datetime.now() 
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "max_results": 20,
                "date_range": ["2023-01-01", "2025-01-20"],
                "min_citations": 10,
                "recency_weight": 0.7,
                "venues": ["CVPR", "ICCV", "NeurIPS"],
                "exclude_seen_papers": True,
                "seen_paper_ids": ["arxiv:2301.12345", "arxiv:2302.67890"],
                "created_at": "2025-01-15T10:00:00",
                "last_updated": "2025-01-20T14:30:00"
            }
        }


# Resolve forward reference to Query for Pydantic
from magpie.models.query import Query
UserProfile.model_rebuild()
