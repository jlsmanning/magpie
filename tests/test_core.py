"""
Tests for Magpie core processing components.

Tests cover query_processor.py, reranker.py, synthesizer.py,
utils/profile_manager.py, and input_manager.py.
"""

import datetime
import os
import tempfile
import pytest
import numpy as np

from magpie.models.query import SubQuery, Query
from magpie.models.profile import UserProfile
from magpie.models.paper import Paper
from magpie.models.results import PaperResult, SearchResults


# =============================================================================
# Query Processor Tests
# =============================================================================

class TestProcessedQuery:
    """Tests for ProcessedQuery dataclass."""

    def test_valid_processed_query(self):
        """Test creating a valid ProcessedQuery."""
        from magpie.core.query_processor import ProcessedQuery

        query = Query(queries=[SubQuery(text="test", weight=1.0)])
        embeddings = [np.array([0.1, 0.2, 0.3])]

        processed = ProcessedQuery(query=query, embeddings=embeddings)
        assert len(processed.embeddings) == 1
        assert processed.query == query

    def test_processed_query_length_mismatch_raises(self):
        """Test that mismatched embeddings count raises error."""
        from magpie.core.query_processor import ProcessedQuery

        query = Query(queries=[
            SubQuery(text="a", weight=0.5),
            SubQuery(text="b", weight=0.5)
        ])
        # Only one embedding for two subqueries
        embeddings = [np.array([0.1, 0.2, 0.3])]

        with pytest.raises(ValueError) as exc_info:
            ProcessedQuery(query=query, embeddings=embeddings)
        assert "must match" in str(exc_info.value)

    def test_processed_query_index_correspondence(self):
        """Test that embeddings correspond to subqueries by index."""
        from magpie.core.query_processor import ProcessedQuery

        query = Query(queries=[
            SubQuery(text="ML", weight=0.6),
            SubQuery(text="CV", weight=0.4)
        ])
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])

        processed = ProcessedQuery(query=query, embeddings=[emb1, emb2])

        # Verify index correspondence
        assert np.array_equal(processed.embeddings[0], emb1)
        assert np.array_equal(processed.embeddings[1], emb2)
        assert processed.query.queries[0].text == "ML"
        assert processed.query.queries[1].text == "CV"


# =============================================================================
# Reranker Tests
# =============================================================================

class TestRerankerBonusCalculations:
    """Tests for reranker bonus calculation functions."""

    def test_recency_bonus_recent_paper(self):
        """Test recency bonus for recent paper."""
        from magpie.core.reranker import _calculate_recency_bonus

        # Paper published today
        recent_date = datetime.date.today()
        bonus = _calculate_recency_bonus(recent_date, recency_weight=1.0)

        # Should get maximum bonus (close to 0.2)
        assert bonus > 0.15
        assert bonus <= 0.2

    def test_recency_bonus_old_paper(self):
        """Test recency bonus for old paper."""
        from magpie.core.reranker import _calculate_recency_bonus

        # Paper published 3 years ago
        old_date = datetime.date.today() - datetime.timedelta(days=1095)
        bonus = _calculate_recency_bonus(old_date, recency_weight=1.0)

        # Should get minimal or no bonus
        assert bonus >= 0.0
        assert bonus < 0.1

    def test_recency_bonus_zero_weight(self):
        """Test recency bonus with zero weight."""
        from magpie.core.reranker import _calculate_recency_bonus

        recent_date = datetime.date.today()
        bonus = _calculate_recency_bonus(recent_date, recency_weight=0.0)

        assert bonus == 0.0

    def test_citation_bonus_high_citations(self):
        """Test citation bonus for highly-cited paper."""
        from magpie.core.reranker import _calculate_citation_bonus

        bonus = _calculate_citation_bonus(citation_count=1000, min_citations=None)

        # Should get good bonus (up to 0.15)
        assert bonus > 0.1
        assert bonus <= 0.15

    def test_citation_bonus_no_citations(self):
        """Test citation bonus with no citation data."""
        from magpie.core.reranker import _calculate_citation_bonus

        bonus = _calculate_citation_bonus(citation_count=None, min_citations=None)
        assert bonus == 0.0

    def test_citation_penalty_below_minimum(self):
        """Test citation penalty when below minimum."""
        from magpie.core.reranker import _calculate_citation_bonus

        bonus = _calculate_citation_bonus(citation_count=5, min_citations=50)

        # Should get penalty
        assert bonus == -0.1

    def test_venue_bonus_preferred_venue(self):
        """Test venue bonus for preferred venue."""
        from magpie.core.reranker import _calculate_venue_bonus

        bonus = _calculate_venue_bonus(
            venue="CVPR 2024",
            preferred_venues=["CVPR", "NeurIPS", "ICLR"]
        )

        assert bonus == 0.1

    def test_venue_bonus_no_match(self):
        """Test venue bonus with no venue match."""
        from magpie.core.reranker import _calculate_venue_bonus

        bonus = _calculate_venue_bonus(
            venue="Some Workshop",
            preferred_venues=["CVPR", "NeurIPS"]
        )

        assert bonus == 0.0

    def test_venue_bonus_no_preferences(self):
        """Test venue bonus with no preferences."""
        from magpie.core.reranker import _calculate_venue_bonus

        bonus = _calculate_venue_bonus(venue="CVPR 2024", preferred_venues=None)
        assert bonus == 0.0


# =============================================================================
# Profile Manager Tests
# =============================================================================

class TestProfileManager:
    """Tests for profile manager functions."""

    @pytest.fixture
    def temp_profile_dir(self, monkeypatch):
        """Create a temporary directory for profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch the Config to use temp directory
            from magpie.utils import config
            monkeypatch.setattr(config.Config, 'PROFILE_DIR', tmpdir)
            yield tmpdir

    def test_create_profile(self, temp_profile_dir):
        """Test creating a new profile."""
        from magpie.utils.profile_manager import create_profile

        profile = create_profile("test_user")

        assert profile.user_id == "test_user"
        assert profile.max_results == 10  # default

    def test_save_and_load_profile(self, temp_profile_dir):
        """Test saving and loading a profile."""
        from magpie.utils.profile_manager import save_profile, load_profile

        # Create and save profile
        profile = UserProfile(
            user_id="test_user",
            max_results=25,
            research_context="Testing ML systems"
        )
        save_profile(profile)

        # Load it back
        loaded = load_profile("test_user")

        assert loaded.user_id == "test_user"
        assert loaded.max_results == 25
        assert loaded.research_context == "Testing ML systems"

    def test_load_nonexistent_creates_new(self, temp_profile_dir):
        """Test that loading nonexistent profile creates a new one."""
        from magpie.utils.profile_manager import load_profile

        profile = load_profile("new_user")

        assert profile.user_id == "new_user"
        assert profile.max_results == 10  # default

    def test_delete_profile(self, temp_profile_dir):
        """Test deleting a profile."""
        from magpie.utils.profile_manager import (
            create_profile, save_profile, delete_profile, profile_exists
        )

        # Create and save
        profile = create_profile("delete_me")
        save_profile(profile)

        assert profile_exists("delete_me") is True

        # Delete
        result = delete_profile("delete_me")

        assert result is True
        assert profile_exists("delete_me") is False

    def test_delete_nonexistent_profile(self, temp_profile_dir):
        """Test deleting a profile that doesn't exist."""
        from magpie.utils.profile_manager import delete_profile

        result = delete_profile("nonexistent")
        assert result is False

    def test_list_profiles(self, temp_profile_dir):
        """Test listing all profiles."""
        from magpie.utils.profile_manager import (
            create_profile, save_profile, list_profiles
        )

        # Create several profiles
        for name in ["user1", "user2", "user3"]:
            save_profile(create_profile(name))

        profiles = list_profiles()

        assert len(profiles) == 3
        assert set(profiles) == {"user1", "user2", "user3"}


# =============================================================================
# Input Manager Tests
# =============================================================================

class TestInputManagerHelpers:
    """Tests for input manager helper functions."""

    def test_format_current_query_none(self):
        """Test formatting when no query exists."""
        from magpie.core.input_manager import _format_current_query

        result = _format_current_query(None)
        assert "No query" in result

    def test_format_current_query_with_topics(self):
        """Test formatting query with topics."""
        from magpie.core.input_manager import _format_current_query

        query = Query(queries=[
            SubQuery(text="machine learning", weight=0.6),
            SubQuery(text="computer vision", weight=0.4)
        ])

        result = _format_current_query(query)

        assert "machine learning" in result
        assert "computer vision" in result
        assert "60%" in result
        assert "40%" in result

    def test_rebalance_weights(self):
        """Test weight rebalancing."""
        from magpie.core.input_manager import _rebalance_weights

        query = Query(queries=[
            SubQuery(text="a", weight=1.0),
        ])
        # Add more subqueries manually (bypassing validation)
        query.queries.append(SubQuery(text="b", weight=0.0))
        query.queries.append(SubQuery(text="c", weight=0.0))

        _rebalance_weights(query)

        # All should be equal now
        for sq in query.queries:
            assert abs(sq.weight - 1/3) < 0.01

    def test_add_subquery_creates_query(self):
        """Test that add_subquery creates query if none exists."""
        from magpie.core.input_manager import _add_subquery

        profile = UserProfile(user_id="test")
        assert profile.current_query is None

        _add_subquery(profile, "machine learning")

        assert profile.current_query is not None
        assert len(profile.current_query.queries) == 1
        assert profile.current_query.queries[0].text == "machine learning"

    def test_add_subquery_to_existing(self):
        """Test adding subquery to existing query."""
        from magpie.core.input_manager import _add_subquery

        profile = UserProfile(user_id="test")
        _add_subquery(profile, "machine learning")
        _add_subquery(profile, "computer vision")

        assert len(profile.current_query.queries) == 2
        # Weights should be rebalanced
        assert abs(profile.current_query.queries[0].weight - 0.5) < 0.01
        assert abs(profile.current_query.queries[1].weight - 0.5) < 0.01

    def test_remove_subquery(self):
        """Test removing a subquery."""
        from magpie.core.input_manager import _add_subquery, _remove_subquery

        profile = UserProfile(user_id="test")
        _add_subquery(profile, "machine learning")
        _add_subquery(profile, "computer vision")

        _remove_subquery(profile, "machine")

        assert len(profile.current_query.queries) == 1
        assert profile.current_query.queries[0].text == "computer vision"
        assert profile.current_query.queries[0].weight == 1.0


# =============================================================================
# Synthesizer Tests
# =============================================================================

class TestSynthesizerBasicExplanation:
    """Tests for basic explanation generation."""

    @pytest.fixture
    def sample_paper(self):
        """Fixture providing a sample paper."""
        return Paper(
            paper_id=("arxiv", "2301.12345"),
            title="Test Paper on Neural Networks",
            authors=["Author One", "Author Two"],
            abstract="This paper studies neural networks.",
            published_date=datetime.date.today() - datetime.timedelta(days=30),
            url="https://arxiv.org/abs/2301.12345",
            venue="NeurIPS 2024",
            categories=["cs.LG", "cs.AI"],
            citation_count=150
        )

    @pytest.fixture
    def sample_profile(self):
        """Fixture providing a sample profile."""
        return UserProfile(user_id="test")

    def test_basic_explanation_single_match(self, sample_paper, sample_profile):
        """Test basic explanation for single query match."""
        from magpie.core.synthesizer import _generate_basic_explanation

        sq = SubQuery(text="neural networks", weight=1.0)
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq],
            relevance_score=0.85
        )

        explanation = _generate_basic_explanation(result, sample_profile)

        assert "neural networks" in explanation
        assert "NeurIPS" in explanation

    def test_basic_explanation_multiple_matches(self, sample_paper, sample_profile):
        """Test basic explanation for multiple query matches."""
        from magpie.core.synthesizer import _generate_basic_explanation

        sq1 = SubQuery(text="neural networks", weight=0.5)
        sq2 = SubQuery(text="deep learning", weight=0.5)
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq1, sq2],
            relevance_score=0.95
        )

        explanation = _generate_basic_explanation(result, sample_profile)

        assert "multiple interests" in explanation.lower()

    def test_basic_explanation_highly_cited(self, sample_paper, sample_profile):
        """Test that highly cited papers are noted."""
        from magpie.core.synthesizer import _generate_basic_explanation

        sq = SubQuery(text="test", weight=1.0)
        result = PaperResult(
            paper=sample_paper,
            matched_subqueries=[sq],
            relevance_score=0.85
        )

        explanation = _generate_basic_explanation(result, sample_profile)

        assert "150 citations" in explanation or "Highly cited" in explanation

    def test_basic_explanation_recent(self, sample_profile):
        """Test that recent papers are noted."""
        from magpie.core.synthesizer import _generate_basic_explanation

        recent_paper = Paper(
            paper_id=("arxiv", "recent"),
            title="Recent Paper",
            authors=["Author"],
            abstract="Abstract",
            published_date=datetime.date.today() - datetime.timedelta(days=10),
            url="https://example.com"
        )
        sq = SubQuery(text="test", weight=1.0)
        result = PaperResult(
            paper=recent_paper,
            matched_subqueries=[sq],
            relevance_score=0.85
        )

        explanation = _generate_basic_explanation(result, sample_profile)

        assert "Recent" in explanation


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration."""

    def test_get_embedding_dim_known_model(self):
        """Test getting embedding dimension for known models."""
        from magpie.utils.config import Config

        assert Config.get_embedding_dim("all-mpnet-base-v2") == 768
        assert Config.get_embedding_dim("all-MiniLM-L6-v2") == 384

    def test_get_embedding_dim_unknown_model(self):
        """Test getting embedding dimension for unknown model defaults to 768."""
        from magpie.utils.config import Config

        assert Config.get_embedding_dim("unknown-model") == 768

    def test_default_values(self):
        """Test that config has expected default values."""
        from magpie.utils.config import Config

        assert Config.DEFAULT_MAX_RESULTS == 10
        assert Config.DEFAULT_RECENCY_WEIGHT == 0.5
        assert Config.VECTOR_DB_COLLECTION == "papers"


# =============================================================================
# Conversation Response Tests
# =============================================================================

class TestConversationResponse:
    """Tests for ConversationResponse class."""

    def test_conversation_response_basic(self):
        """Test basic ConversationResponse."""
        from magpie.core.input_manager import ConversationResponse

        response = ConversationResponse(message_to_user="Hello!")

        assert response.message_to_user == "Hello!"
        assert response.has_query() is False
        assert response.has_profile_updates() is False

    def test_conversation_response_with_query(self):
        """Test ConversationResponse with query."""
        from magpie.core.input_manager import ConversationResponse

        query = Query(queries=[SubQuery(text="test", weight=1.0)])
        response = ConversationResponse(
            message_to_user="Searching...",
            query=query
        )

        assert response.has_query() is True
        assert response.query == query

    def test_conversation_response_with_updates(self):
        """Test ConversationResponse with profile updates."""
        from magpie.core.input_manager import ConversationResponse

        response = ConversationResponse(
            message_to_user="Updated!",
            profile_updates={"modified": True}
        )

        assert response.has_profile_updates() is True
        assert response.profile_updates["modified"] is True
