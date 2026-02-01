"""
Tests for Magpie data models.

Tests cover query.py, profile.py, paper.py, and results.py.
"""

import datetime
import pytest
import pydantic

from magpie.models.query import SubQuery, Query
from magpie.models.profile import UserProfile
from magpie.models.paper import Paper
from magpie.models.results import PaperResult, SearchResults


# =============================================================================
# SubQuery Tests
# =============================================================================

class TestSubQuery:
    """Tests for SubQuery model."""

    def test_valid_subquery(self):
        """Test creating a valid SubQuery."""
        sq = SubQuery(text="machine learning", weight=0.5)
        assert sq.text == "machine learning"
        assert sq.weight == 0.5
        assert sq.source_interest_ids is None

    def test_subquery_with_interest_ids(self):
        """Test SubQuery with source interest IDs."""
        sq = SubQuery(
            text="computer vision",
            weight=0.3,
            source_interest_ids=["id-1", "id-2"]
        )
        assert sq.source_interest_ids == ["id-1", "id-2"]

    def test_subquery_text_stripped(self):
        """Test that text is stripped of whitespace."""
        sq = SubQuery(text="  neural networks  ", weight=0.5)
        assert sq.text == "neural networks"

    def test_subquery_empty_text_raises(self):
        """Test that empty text raises validation error."""
        with pytest.raises(pydantic.ValidationError):
            SubQuery(text="", weight=0.5)

    def test_subquery_whitespace_text_raises(self):
        """Test that whitespace-only text raises validation error."""
        with pytest.raises(pydantic.ValidationError):
            SubQuery(text="   ", weight=0.5)

    def test_subquery_weight_below_zero_raises(self):
        """Test that weight below 0 raises validation error."""
        with pytest.raises(pydantic.ValidationError):
            SubQuery(text="test", weight=-0.1)

    def test_subquery_weight_above_one_raises(self):
        """Test that weight above 1 raises validation error."""
        with pytest.raises(pydantic.ValidationError):
            SubQuery(text="test", weight=1.1)

    def test_subquery_weight_boundaries(self):
        """Test weight at boundary values."""
        sq_zero = SubQuery(text="test", weight=0.0)
        sq_one = SubQuery(text="test", weight=1.0)
        assert sq_zero.weight == 0.0
        assert sq_one.weight == 1.0


# =============================================================================
# Query Tests
# =============================================================================

class TestQuery:
    """Tests for Query model."""

    def test_valid_query(self):
        """Test creating a valid Query."""
        query = Query(
            queries=[SubQuery(text="deep learning", weight=1.0)],
            max_results=20
        )
        assert len(query.queries) == 1
        assert query.max_results == 20
        assert query.recency_weight == 0.5  # default

    def test_query_multiple_subqueries(self):
        """Test Query with multiple subqueries summing to 1."""
        query = Query(
            queries=[
                SubQuery(text="NLP", weight=0.6),
                SubQuery(text="transformers", weight=0.4)
            ]
        )
        assert len(query.queries) == 2

    def test_query_weights_must_sum_to_one(self):
        """Test that subquery weights must sum to 1.0."""
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Query(
                queries=[
                    SubQuery(text="NLP", weight=0.5),
                    SubQuery(text="transformers", weight=0.3)
                ]
            )
        assert "sum to 1.0" in str(exc_info.value)

    def test_query_weights_allow_small_error(self):
        """Test that small floating point errors are allowed."""
        query = Query(
            queries=[
                SubQuery(text="a", weight=0.333),
                SubQuery(text="b", weight=0.333),
                SubQuery(text="c", weight=0.334)
            ]
        )
        assert len(query.queries) == 3

    def test_query_empty_queries_raises(self):
        """Test that empty queries list raises validation error."""
        with pytest.raises(pydantic.ValidationError):
            Query(queries=[])

    def test_query_date_range_valid(self):
        """Test valid date range."""
        query = Query(
            queries=[SubQuery(text="test", weight=1.0)],
            date_range=(datetime.date(2023, 1, 1), datetime.date(2024, 1, 1))
        )
        assert query.date_range[0] == datetime.date(2023, 1, 1)

    def test_query_date_range_invalid_order(self):
        """Test that start date after end date raises error."""
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Query(
                queries=[SubQuery(text="test", weight=1.0)],
                date_range=(datetime.date(2024, 1, 1), datetime.date(2023, 1, 1))
            )
        assert "must be before" in str(exc_info.value)

    def test_query_max_results_boundaries(self):
        """Test max_results boundaries."""
        query_min = Query(
            queries=[SubQuery(text="test", weight=1.0)],
            max_results=1
        )
        query_max = Query(
            queries=[SubQuery(text="test", weight=1.0)],
            max_results=100
        )
        assert query_min.max_results == 1
        assert query_max.max_results == 100

    def test_query_max_results_out_of_bounds(self):
        """Test max_results out of bounds raises error."""
        with pytest.raises(pydantic.ValidationError):
            Query(
                queries=[SubQuery(text="test", weight=1.0)],
                max_results=0
            )
        with pytest.raises(pydantic.ValidationError):
            Query(
                queries=[SubQuery(text="test", weight=1.0)],
                max_results=101
            )

    def test_query_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        query = Query(queries=[SubQuery(text="test", weight=1.0)])
        assert query.timestamp is not None
        assert isinstance(query.timestamp, datetime.datetime)


# =============================================================================
# UserProfile Tests
# =============================================================================

class TestUserProfile:
    """Tests for UserProfile model."""

    def test_valid_user_profile(self):
        """Test creating a valid UserProfile."""
        profile = UserProfile(user_id="user-1")
        assert profile.user_id == "user-1"
        assert profile.max_results == 10  # default

    def test_user_profile_defaults(self):
        """Test UserProfile default values."""
        profile = UserProfile(user_id="user-1")
        assert profile.max_results == 10
        assert profile.recency_weight == 0.5
        assert profile.exclude_seen_papers is True
        assert profile.date_range is None
        assert profile.min_citations is None
        assert profile.venues is None
        assert profile.current_query is None
        assert profile.research_context is None
        assert profile.seen_papers == {}

    def test_user_profile_with_research_context(self):
        """Test UserProfile with research context."""
        profile = UserProfile(
            user_id="user-1",
            research_context="Studying explainable AI for medical imaging"
        )
        assert profile.research_context == "Studying explainable AI for medical imaging"

    def test_user_profile_date_range_invalid(self):
        """Test that invalid date range raises error."""
        with pytest.raises(pydantic.ValidationError):
            UserProfile(
                user_id="user-1",
                date_range=(datetime.date(2024, 1, 1), datetime.date(2023, 1, 1))
            )

    def test_user_profile_date_range_valid(self):
        """Test valid date range on profile."""
        profile = UserProfile(
            user_id="user-1",
            date_range=(datetime.date(2023, 1, 1), datetime.date(2024, 1, 1))
        )
        assert profile.date_range == (datetime.date(2023, 1, 1), datetime.date(2024, 1, 1))

    def test_mark_paper_seen(self):
        """Test mark_paper_seen method."""
        profile = UserProfile(user_id="user-1")
        profile.mark_paper_seen("arxiv:1234")

        assert "arxiv:1234" in profile.seen_papers
        assert isinstance(profile.seen_papers["arxiv:1234"], datetime.datetime)

    def test_has_seen_paper(self):
        """Test has_seen_paper method."""
        profile = UserProfile(user_id="user-1")
        profile.mark_paper_seen("arxiv:1234")

        assert profile.has_seen_paper("arxiv:1234") is True
        assert profile.has_seen_paper("arxiv:5678") is False

    def test_cleanup_old_seen_papers(self):
        """Test cleanup_old_seen_papers method."""
        profile = UserProfile(user_id="user-1")

        # Add a recent paper
        profile.seen_papers["recent"] = datetime.datetime.now()
        # Add an old paper (100 days ago)
        profile.seen_papers["old"] = datetime.datetime.now() - datetime.timedelta(days=100)

        profile.cleanup_old_seen_papers(days=90)

        assert "recent" in profile.seen_papers
        assert "old" not in profile.seen_papers

    def test_user_profile_timestamps(self):
        """Test that created_at and last_updated are auto-set."""
        profile = UserProfile(user_id="user-1")
        assert profile.created_at is not None
        assert profile.last_updated is not None
        assert isinstance(profile.created_at, datetime.datetime)
        assert isinstance(profile.last_updated, datetime.datetime)

    def test_user_profile_max_results_boundaries(self):
        """Test max_results boundaries on profile."""
        profile_min = UserProfile(user_id="user-1", max_results=1)
        profile_max = UserProfile(user_id="user-2", max_results=100)
        assert profile_min.max_results == 1
        assert profile_max.max_results == 100

    def test_user_profile_max_results_out_of_bounds(self):
        """Test max_results out of bounds raises error."""
        with pytest.raises(pydantic.ValidationError):
            UserProfile(user_id="user-1", max_results=0)
        with pytest.raises(pydantic.ValidationError):
            UserProfile(user_id="user-1", max_results=101)

    def test_user_profile_recency_weight_boundaries(self):
        """Test recency_weight boundaries."""
        profile_min = UserProfile(user_id="user-1", recency_weight=0.0)
        profile_max = UserProfile(user_id="user-2", recency_weight=1.0)
        assert profile_min.recency_weight == 0.0
        assert profile_max.recency_weight == 1.0

    def test_user_profile_recency_weight_out_of_bounds(self):
        """Test recency_weight out of bounds raises error."""
        with pytest.raises(pydantic.ValidationError):
            UserProfile(user_id="user-1", recency_weight=-0.1)
        with pytest.raises(pydantic.ValidationError):
            UserProfile(user_id="user-1", recency_weight=1.1)


# =============================================================================
# Paper Tests
# =============================================================================

class TestPaper:
    """Tests for Paper model."""

    @pytest.fixture
    def valid_paper_data(self):
        """Fixture providing valid paper data."""
        return {
            "paper_id": ("arxiv", "2301.12345"),
            "title": "A Study on Neural Networks",
            "authors": ["Jane Doe", "John Smith"],
            "abstract": "This paper presents a novel approach...",
            "published_date": datetime.date(2024, 1, 15),
            "url": "https://arxiv.org/abs/2301.12345"
        }

    def test_valid_paper(self, valid_paper_data):
        """Test creating a valid Paper."""
        paper = Paper(**valid_paper_data)
        assert paper.title == "A Study on Neural Networks"
        assert paper.source == "arxiv"
        assert paper.id == "2301.12345"

    def test_paper_optional_fields(self, valid_paper_data):
        """Test Paper with optional fields."""
        paper = Paper(
            **valid_paper_data,
            pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
            doi="10.48550/arXiv.2301.12345",
            categories=["cs.CV", "cs.AI"],
            citation_count=42,
            venue="CVPR 2024"
        )
        assert paper.pdf_url is not None
        assert paper.citation_count == 42

    def test_paper_title_stripped(self, valid_paper_data):
        """Test that title is stripped of whitespace."""
        valid_paper_data["title"] = "  Spaced Title  "
        paper = Paper(**valid_paper_data)
        assert paper.title == "Spaced Title"

    def test_paper_empty_title_raises(self, valid_paper_data):
        """Test that empty title raises validation error."""
        valid_paper_data["title"] = ""
        with pytest.raises(pydantic.ValidationError):
            Paper(**valid_paper_data)

    def test_paper_whitespace_title_raises(self, valid_paper_data):
        """Test that whitespace-only title raises validation error."""
        valid_paper_data["title"] = "   "
        with pytest.raises(pydantic.ValidationError):
            Paper(**valid_paper_data)

    def test_paper_empty_abstract_raises(self, valid_paper_data):
        """Test that empty abstract raises validation error."""
        valid_paper_data["abstract"] = ""
        with pytest.raises(pydantic.ValidationError):
            Paper(**valid_paper_data)

    def test_paper_empty_authors_raises(self, valid_paper_data):
        """Test that empty authors list raises validation error."""
        valid_paper_data["authors"] = []
        with pytest.raises(pydantic.ValidationError):
            Paper(**valid_paper_data)

    def test_paper_source_property(self, valid_paper_data):
        """Test source property."""
        paper = Paper(**valid_paper_data)
        assert paper.source == "arxiv"

    def test_paper_id_property(self, valid_paper_data):
        """Test id property."""
        paper = Paper(**valid_paper_data)
        assert paper.id == "2301.12345"

    def test_paper_str_short_authors(self, valid_paper_data):
        """Test __str__ with few authors."""
        paper = Paper(**valid_paper_data)
        result = str(paper)
        assert "A Study on Neural Networks" in result
        assert "Jane Doe" in result
        assert "2024" in result

    def test_paper_str_many_authors(self, valid_paper_data):
        """Test __str__ with many authors shows 'et al.'"""
        valid_paper_data["authors"] = ["A", "B", "C", "D", "E"]
        paper = Paper(**valid_paper_data)
        result = str(paper)
        assert "et al." in result

    def test_paper_citation_count_negative_raises(self, valid_paper_data):
        """Test that negative citation count raises error."""
        with pytest.raises(pydantic.ValidationError):
            Paper(**valid_paper_data, citation_count=-1)

    def test_paper_metadata(self, valid_paper_data):
        """Test paper metadata field."""
        paper = Paper(
            **valid_paper_data,
            metadata={"arxiv_comment": "Accepted at CVPR"}
        )
        assert paper.metadata["arxiv_comment"] == "Accepted at CVPR"


# =============================================================================
# PaperResult Tests
# =============================================================================

class TestPaperResult:
    """Tests for PaperResult model."""

    @pytest.fixture
    def sample_paper(self):
        """Fixture providing a sample paper."""
        return Paper(
            paper_id=("arxiv", "2301.12345"),
            title="Test Paper",
            authors=["Author One"],
            abstract="This is an abstract.",
            published_date=datetime.date(2024, 1, 15),
            url="https://arxiv.org/abs/2301.12345"
        )

    @pytest.fixture
    def sample_subquery(self):
        """Fixture providing a sample subquery."""
        return SubQuery(
            text="machine learning",
            weight=1.0,
            source_interest_ids=["int-1", "int-2"]
        )

    def test_valid_paper_result(self, sample_paper, sample_subquery):
        """Test creating a valid PaperResult."""
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sample_subquery],
            relevance_score=0.85
        )
        assert result.relevance_score == 0.85
        assert result.explanation is None

    def test_paper_result_with_explanation(self, sample_paper, sample_subquery):
        """Test PaperResult with explanation."""
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sample_subquery],
            relevance_score=0.85,
            explanation="Highly relevant to ML interests"
        )
        assert result.explanation == "Highly relevant to ML interests"

    def test_paper_result_empty_subqueries_raises(self, sample_paper):
        """Test that empty matched_subqueries raises error."""
        with pytest.raises(pydantic.ValidationError):
            PaperResult(
                paper=sample_paper,
                matched_subqueries=[],
                relevance_score=0.85
            )

    def test_paper_result_score_boundaries(self, sample_paper, sample_subquery):
        """Test relevance_score boundaries."""
        result_zero = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sample_subquery],
            relevance_score=0.0
        )
        result_one = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sample_subquery],
            relevance_score=1.0
        )
        assert result_zero.relevance_score == 0.0
        assert result_one.relevance_score == 1.0

    def test_paper_result_score_out_of_bounds(self, sample_paper, sample_subquery):
        """Test that score out of bounds raises error."""
        with pytest.raises(pydantic.ValidationError):
            PaperResult(
                paper=sample_paper,
                matched_subqueries=[sample_subquery],
                relevance_score=-0.1
            )
        with pytest.raises(pydantic.ValidationError):
            PaperResult(
                paper=sample_paper,
                matched_subqueries=[sample_subquery],
                relevance_score=1.1
            )

    def test_get_source_interest_ids(self, sample_paper):
        """Test get_source_interest_ids method."""
        sq1 = SubQuery(text="ML", weight=0.5, source_interest_ids=["int-1", "int-2"])
        sq2 = SubQuery(text="CV", weight=0.5, source_interest_ids=["int-2", "int-3"])

        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq1, sq2],
            relevance_score=0.9
        )

        interest_ids = result.get_source_interest_ids()
        assert interest_ids == {"int-1", "int-2", "int-3"}

    def test_get_source_interest_ids_none(self, sample_paper):
        """Test get_source_interest_ids with no interest IDs."""
        sq = SubQuery(text="ML", weight=1.0)  # No source_interest_ids
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq],
            relevance_score=0.9
        )

        interest_ids = result.get_source_interest_ids()
        assert interest_ids == set()

    def test_matched_multiple_queries(self, sample_paper):
        """Test matched_multiple_queries method."""
        sq1 = SubQuery(text="ML", weight=0.5)
        sq2 = SubQuery(text="CV", weight=0.5)

        single = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq1],
            relevance_score=0.9
        )
        multiple = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq1, sq2],
            relevance_score=0.9
        )

        assert single.matched_multiple_queries() is False
        assert multiple.matched_multiple_queries() is True

    def test_paper_result_with_rerank_reasoning(self, sample_paper, sample_subquery):
        """Test PaperResult with rerank reasoning."""
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sample_subquery],
            relevance_score=0.85,
            rerank_reasoning="Strong methodological contribution"
        )
        assert result.rerank_reasoning == "Strong methodological contribution"


# =============================================================================
# SearchResults Tests
# =============================================================================

class TestSearchResults:
    """Tests for SearchResults model."""

    @pytest.fixture
    def sample_query(self):
        """Fixture providing a sample query."""
        return Query(
            queries=[SubQuery(text="machine learning", weight=1.0)],
            max_results=10
        )

    @pytest.fixture
    def sample_paper_result(self):
        """Fixture providing a sample paper result."""
        paper = Paper(
            paper_id=("arxiv", "2301.12345"),
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract text",
            published_date=datetime.date(2024, 1, 15),
            url="https://arxiv.org/abs/2301.12345"
        )
        subquery = SubQuery(
            text="ML",
            weight=1.0,
            source_interest_ids=["int-1"]
        )
        return PaperResult(
            paper=paper,
            matched_subqueries=[subquery],
            relevance_score=0.85
        )

    def test_valid_search_results(self, sample_query, sample_paper_result):
        """Test creating valid SearchResults."""
        results = SearchResults(
            results=[sample_paper_result],
            query=sample_query,
            total_found=100
        )
        assert len(results) == 1
        assert results.total_found == 100

    def test_search_results_empty(self, sample_query):
        """Test SearchResults with no results."""
        results = SearchResults(
            results=[],
            query=sample_query,
            total_found=0
        )
        assert len(results) == 0

    def test_search_results_timestamp(self, sample_query):
        """Test that timestamp is auto-set."""
        results = SearchResults(
            results=[],
            query=sample_query,
            total_found=0
        )
        assert results.timestamp is not None

    def test_get_papers(self, sample_query, sample_paper_result):
        """Test get_papers method."""
        results = SearchResults(
            results=[sample_paper_result],
            query=sample_query,
            total_found=1
        )
        papers = results.get_papers()
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"

    def test_get_paper_ids(self, sample_query, sample_paper_result):
        """Test get_paper_ids method."""
        results = SearchResults(
            results=[sample_paper_result],
            query=sample_query,
            total_found=1
        )
        paper_ids = results.get_paper_ids()
        assert paper_ids == {("arxiv", "2301.12345")}

    def test_get_results_by_interest(self, sample_query):
        """Test get_results_by_interest method."""
        paper1 = Paper(
            paper_id=("arxiv", "1"),
            title="Paper 1",
            authors=["A"],
            abstract="Abstract",
            published_date=datetime.date(2024, 1, 1),
            url="https://example.com/1"
        )
        paper2 = Paper(
            paper_id=("arxiv", "2"),
            title="Paper 2",
            authors=["B"],
            abstract="Abstract",
            published_date=datetime.date(2024, 1, 1),
            url="https://example.com/2"
        )

        sq1 = SubQuery(text="ML", weight=0.5, source_interest_ids=["int-1"])
        sq2 = SubQuery(text="CV", weight=0.5, source_interest_ids=["int-2"])

        pr1 = PaperResult(paper=paper1, matched_subqueries=[sq1], relevance_score=0.9)
        pr2 = PaperResult(paper=paper2, matched_subqueries=[sq2], relevance_score=0.8)

        results = SearchResults(
            results=[pr1, pr2],
            query=sample_query,
            total_found=2
        )

        int1_results = results.get_results_by_interest("int-1")
        assert len(int1_results) == 1
        assert int1_results[0].paper.title == "Paper 1"

    def test_get_multi_query_matches(self, sample_query):
        """Test get_multi_query_matches method."""
        paper = Paper(
            paper_id=("arxiv", "1"),
            title="Multi-match Paper",
            authors=["A"],
            abstract="Abstract",
            published_date=datetime.date(2024, 1, 1),
            url="https://example.com/1"
        )

        sq1 = SubQuery(text="ML", weight=0.5)
        sq2 = SubQuery(text="CV", weight=0.5)

        pr_single = PaperResult(paper=paper, matched_subqueries=[sq1], relevance_score=0.9)
        pr_multi = PaperResult(paper=paper, matched_subqueries=[sq1, sq2], relevance_score=0.95)

        results = SearchResults(
            results=[pr_single, pr_multi],
            query=sample_query,
            total_found=2
        )

        multi_matches = results.get_multi_query_matches()
        assert len(multi_matches) == 1
        assert multi_matches[0].relevance_score == 0.95

    def test_len(self, sample_query, sample_paper_result):
        """Test __len__ method."""
        results = SearchResults(
            results=[sample_paper_result, sample_paper_result],
            query=sample_query,
            total_found=2
        )
        assert len(results) == 2
