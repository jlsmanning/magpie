"""
User profile data models for Magpie.

Defines user research interests and default search preferences.
"""

import datetime
import pydantic
import typing

class Interest(pydantic.BaseModel):
    """
    A research interest/topic that the user wants to track.
    
    weight_stars determines relative importance - higher stars means
    this interest will receive proportionally more papers in search results.
    Stars are normalized to weights when generating queries.
    """
    id: str = pydantic.Field(
        ...,
        description="Unique identifier for this interest"
    ) #FIXME: ensure unique in InputParserManager
    
    topic: str = pydantic.Field(
        ...,
        description="The research interest (e.g., 'explainable AI', 'computer vision')",
        min_length=1
    ) #FIXME: ensure no duplicates
    
    weight_stars: int = pydantic.Field(
        ...,
        description="Priority rating - higher = more important (1-100)",
        ge=1,
        le=100
    )
    
    added_date: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now,
        description="When this interest was added"
    )
    
    last_modified: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now,
        description="When this interest was last updated"
    )
    
    @pydantic.field_validator('topic')
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        """Ensure topic is not just whitespace."""
        if not v.strip():
            raise ValueError("Topic cannot be empty or whitespace")
        return v.strip()


class UserProfile(pydantic.BaseModel):
    """
    Complete user profile containing research interests and search preferences.
    
    Interests are stored with star ratings that get normalized to weights
    when generating queries. Default search parameters can be overridden
    per-query.
    """
    user_id: str = pydantic.Field(
        ...,
        description="Unique identifier for this user"
    ) #FIXME: ensure unique
    
    interests: typing.List[Interest] = pydantic.Field(
        default_factory=list,
        description="typing.List of research interests"
    ) #FIXME: ensure is set
    
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
    
    # History tracking
    seen_papers: Dict[str, datetime.datetime] = pydantic.Field(
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
    def validate_date_range(cls, v: typing.Optional[typing.Tuple[datetime.date, datetime.date]]) -> typing.Optional[typing.Tuple[date, date]]:
        """Ensure start date is before end date."""
        if v is not None:
            start_date, end_date = v
            if start_date > end_date:
                raise ValueError(f"Start date {start_date} must be before end date {end_date}")
        return v

    @pydantic.field_validator('interests')
    @classmethod
    def no_duplicate_topics(cls, v: List[Interest]) -> List[Interest]:
        topics_lower = [interest.topic.lower() for interest in v]
        if len(topics_lower) != len(set(topics_lower)):
            raise ValueError("Cannot have multiple interests with the same topic")
        return v
    
    def get_interest_by_id(self, interest_id: str) -> typing.Optional[Interest]:
        """Find an interest by its ID."""
        for interest in self.interests:
            if interest.id == interest_id:
                return interest
        return None
    
    def add_interest(self, interest: Interest) -> None:
        """Add a new interest to the profile."""
        # Check for duplicate IDs
        if self.get_interest_by_id(interest.id) is not None:
            raise ValueError(f"Interest with ID {interest.id} already exists")
        if any(i.topic.lower() == interest.topic.lower() for i in self.interests):
            raise ValueError(f"Interest with topic '{interest.topic}' already exists")
        self.interests.append(interest)
        self.last_updated = datetime.datetime.now()
    
    def remove_interest(self, interest_id: str) -> bool:
        """
        Remove an interest by ID.
        Returns True if removed, False if not found.
        """
        for i, interest in enumerate(self.interests):
            if interest.id == interest_id:
                self.interests.pop(i)
                self.last_updated = datetime.datetime.now()
                return True
        return False
    
    def update_interest_stars(self, interest_id: str, new_stars: int) -> bool:
        """
        Update the star rating for an interest.
        Returns True if updated, False if not found.
        """
        interest = self.get_interest_by_id(interest_id)
        if interest is not None:
            interest.weight_stars = new_stars
            interest.last_modified = datetime.datetime.now()
            self.last_updated = datetime.datetime.now()
            return True
        return False
    
    def normalize_interest_weights(self) -> dict[str, float]:
        """
        Convert star ratings to normalized weights that sum to 1.0.
        Returns dict mapping interest_id -> weight.
        """
        if not self.interests:
            return {}
        
        total_stars = sum(interest.weight_stars for interest in self.interests)
        return {
            interest.id: interest.weight_stars / total_stars
            for interest in self.interests
        }

    def has_topic(self, topic: str) -> bool:
        """Check if an interest with this topic already exists."""
        return any(
            interest.topic.lower() == topic.lower() 
            for interest in self.interests
        )

    def get_interests_by_topic(self, topic: str) -> typing.List[Interest]:
        """
        Find all interests containing the given topic substring (case-insensitive).

        Example: search for "vision" returns interests with topics like
        "computer vision", "vision transformers", "3D vision", etc.
        """
        search_term = topic.lower()
        return [
            interest for interest in self.interests
            if search_term in interest.topic.lower()
        ]
    
    def mark_paper_seen(self, paper_id: str) -> None:
        """Mark a paper as seen."""
        self.seen_papers[paper_id] = datetime.datetime.now()
        self.last_updated = datetime.datetime.now()
    
    def has_seen_paper(self, paper_id: str) -> bool:
        """Check if user has already seen a paper."""
        return paper_id in self.seen_papers

    def cleanup_old_seen_papers(self, days: int = 90) -> None: #FIXME: ensure I'm actually using this!
        """Remove papers seen more than N days ago."""
        cutoff = datetime.now() - datetime.timedelta(days=days)
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
                "interests": [
                    {
                        "id": "interest-uuid-1",
                        "topic": "explainable AI",
                        "weight_stars": 6,
                        "added_date": "2025-01-15T10:00:00",
                        "last_modified": "2025-01-20T14:30:00"
                    },
                    {
                        "id": "interest-uuid-2",
                        "topic": "computer vision bias",
                        "weight_stars": 4,
                        "added_date": "2025-01-15T10:00:00",
                        "last_modified": "2025-01-15T10:00:00"
                    },
                    {
                        "id": "interest-uuid-3",
                        "topic": "fine-grained classification",
                        "weight_stars": 2,
                        "added_date": "2025-01-18T09:00:00",
                        "last_modified": "2025-01-18T09:00:00"
                    }
                ],
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
