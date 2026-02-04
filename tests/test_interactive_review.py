"""
Tests for interactive review session.

Tests cover InteractiveReviewSession class and its methods.
"""

import datetime
import pytest
from unittest.mock import Mock, patch, MagicMock

from magpie.models.query import SubQuery, Query
from magpie.models.paper import Paper
from magpie.models.results import PaperResult, SearchResults


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        Paper(
            paper_id=("arxiv", "2301.00001"),
            title="Paper One: Machine Learning",
            authors=["Author A", "Author B"],
            abstract="This paper explores machine learning techniques.",
            published_date=datetime.date(2024, 1, 15),
            url="https://arxiv.org/abs/2301.00001",
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf"
        ),
        Paper(
            paper_id=("arxiv", "2301.00002"),
            title="Paper Two: Computer Vision",
            authors=["Author C"],
            abstract="This paper explores computer vision methods.",
            published_date=datetime.date(2024, 2, 20),
            url="https://arxiv.org/abs/2301.00002",
            pdf_url="https://arxiv.org/pdf/2301.00002.pdf"
        ),
        Paper(
            paper_id=("arxiv", "2301.00003"),
            title="Paper Three: NLP",
            authors=["Author D", "Author E"],
            abstract="This paper explores natural language processing.",
            published_date=datetime.date(2024, 3, 10),
            url="https://arxiv.org/abs/2301.00003"
            # No pdf_url for this one
        ),
    ]


@pytest.fixture
def sample_search_results(sample_papers):
    """Create sample search results for testing."""
    sq = SubQuery(text="machine learning", weight=1.0)
    query = Query(queries=[sq])

    results = []
    for paper in sample_papers:
        results.append(PaperResult(
            paper=paper,
            matched_subqueries=[sq],
            relevance_score=0.85,
            explanation=f"This paper is relevant because it discusses {paper.title.split(':')[1].strip().lower()}."
        ))

    return SearchResults(
        results=results,
        query=query,
        total_found=3
    )


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    mock = Mock()
    mock.chat_with_json = Mock(return_value={
        "action": "present_paper",
        "message": "Here's the paper..."
    })
    mock.chat = Mock(return_value="This is an answer about the paper.")
    return mock


@pytest.fixture
def mock_pdf_fetcher():
    """Create mock PDF fetcher."""
    mock = Mock()
    mock.fetch_and_extract = Mock(return_value=("/tmp/paper.pdf", "Full paper text content here."))
    return mock


@pytest.fixture
def mock_zotero_client():
    """Create mock Zotero client."""
    mock = Mock()
    mock.save_paper = Mock(return_value="ZOTERO_KEY_123")
    return mock


# =============================================================================
# InteractiveReviewSession Tests
# =============================================================================

class TestInteractiveReviewSession:
    """Tests for InteractiveReviewSession."""

    def test_init_with_results(self, sample_search_results, mock_llm_client):
        """Test initializing session with search results."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        assert session.current_index == 0
        assert len(session.results.results) == 3
        assert session.downloaded_pdfs == {}
        assert session.saved_to_zotero == set()

    def test_start_review(self, sample_search_results, mock_llm_client):
        """Test starting a review session."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        message = session.start_review()

        assert "3 papers" in message
        assert "Ready to review" in message

    def test_start_review_empty_results(self, mock_llm_client):
        """Test starting review with no results."""
        from magpie.core.interactive_review import InteractiveReviewSession

        sq = SubQuery(text="test", weight=1.0)
        query = Query(queries=[sq])
        empty_results = SearchResults(results=[], query=query, total_found=0)

        session = InteractiveReviewSession(
            results=empty_results,
            llm_client=mock_llm_client
        )

        message = session.start_review()
        assert "No papers" in message

    def test_present_current_paper(self, sample_search_results, mock_llm_client):
        """Test presenting current paper."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session._present_current_paper()

        assert "Paper 1 of 3" in response
        assert "Paper One: Machine Learning" in response
        assert "machine learning" in response.lower()

    def test_skip_current_paper(self, sample_search_results, mock_llm_client):
        """Test skipping current paper."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        assert session.current_index == 0

        response = session._skip_current_paper()

        assert session.current_index == 1
        assert "paper 2" in response.lower()

    def test_skip_last_paper(self, sample_search_results, mock_llm_client):
        """Test skipping the last paper."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )
        session.current_index = 2  # Last paper

        response = session._skip_current_paper()

        assert "last paper" in response.lower() or "complete" in response.lower()

    def test_switch_to_paper_valid(self, sample_search_results, mock_llm_client):
        """Test switching to a valid paper index."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session._switch_to_paper(2)

        assert session.current_index == 2
        assert "Paper Three" in response

    def test_switch_to_paper_invalid(self, sample_search_results, mock_llm_client):
        """Test switching to invalid paper index."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session._switch_to_paper(10)

        assert "doesn't exist" in response

    def test_switch_to_paper_negative(self, sample_search_results, mock_llm_client):
        """Test switching to negative paper index."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session._switch_to_paper(-1)

        assert "doesn't exist" in response

    def test_exit_review(self, sample_search_results, mock_llm_client):
        """Test exiting review."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )
        session.saved_to_zotero.add("arxiv:2301.00001")

        response = session._exit_review()

        assert "ended" in response.lower()
        assert "1 paper" in response

    def test_save_paper_no_zotero(self, sample_search_results, mock_llm_client):
        """Test saving paper when Zotero is not configured."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client,
            zotero_client=None
        )

        response = session._save_current_paper()

        assert "not configured" in response.lower()

    def test_save_paper_with_zotero(
        self,
        sample_search_results,
        mock_llm_client,
        mock_zotero_client,
        mock_pdf_fetcher
    ):
        """Test saving paper with Zotero configured."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client,
            zotero_client=mock_zotero_client,
            pdf_fetcher=mock_pdf_fetcher
        )

        response = session._save_current_paper()

        assert "Saved" in response
        assert "arxiv:2301.00001" in session.saved_to_zotero
        mock_zotero_client.save_paper.assert_called_once()

    def test_save_paper_already_saved(
        self,
        sample_search_results,
        mock_llm_client,
        mock_zotero_client
    ):
        """Test saving paper that was already saved."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client,
            zotero_client=mock_zotero_client
        )
        session.saved_to_zotero.add("arxiv:2301.00001")

        response = session._save_current_paper()

        assert "already saved" in response.lower()
        mock_zotero_client.save_paper.assert_not_called()

    def test_discuss_paper_without_pdf(self, sample_search_results, mock_llm_client):
        """Test discussing paper without needing PDF."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session._discuss_paper("What is this paper about?", needs_pdf=False)

        assert response == "This is an answer about the paper."
        mock_llm_client.chat.assert_called_once()

    def test_discuss_paper_with_pdf_download(
        self,
        sample_search_results,
        mock_llm_client,
        mock_pdf_fetcher
    ):
        """Test discussing paper that needs PDF download."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client,
            pdf_fetcher=mock_pdf_fetcher
        )

        response = session._discuss_paper("What methods do they use?", needs_pdf=True)

        mock_pdf_fetcher.fetch_and_extract.assert_called_once()
        assert "arxiv:2301.00001" in session.downloaded_pdfs

    def test_discuss_paper_no_pdf_available(self, sample_search_results, mock_llm_client):
        """Test discussing paper when no PDF is available."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )
        session.current_index = 2  # Paper with no pdf_url

        response = session._discuss_paper("What methods?", needs_pdf=True)

        assert "doesn't have a PDF" in response

    def test_get_current_paper(self, sample_search_results, mock_llm_client):
        """Test getting current paper."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        paper = session._get_current_paper()

        assert paper.title == "Paper One: Machine Learning"

        session.current_index = 1
        paper = session._get_current_paper()

        assert paper.title == "Paper Two: Computer Vision"

    def test_add_to_history(self, sample_search_results, mock_llm_client):
        """Test adding messages to conversation history."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        session._add_to_history("Test message")

        assert len(session.conversation_history) == 1
        assert session.conversation_history[0]["role"] == "assistant"
        assert session.conversation_history[0]["content"] == "Test message"
        assert session.last_response == "Test message"

    def test_process_message_present_action(self, sample_search_results, mock_llm_client):
        """Test processing message with present_paper action."""
        from magpie.core.interactive_review import InteractiveReviewSession

        mock_llm_client.chat_with_json.return_value = {
            "action": "present_paper",
            "message": "Here's the paper"
        }

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session.process_message("Show me the paper")

        assert "Paper 1 of 3" in response

    def test_process_message_skip_action(self, sample_search_results, mock_llm_client):
        """Test processing message with skip action."""
        from magpie.core.interactive_review import InteractiveReviewSession

        mock_llm_client.chat_with_json.return_value = {
            "action": "skip_current",
            "message": "Skipping..."
        }

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session.process_message("Skip this one")

        assert session.current_index == 1

    def test_process_message_switch_action_none_index(self, sample_search_results, mock_llm_client):
        """Test processing switch action with None index."""
        from magpie.core.interactive_review import InteractiveReviewSession

        mock_llm_client.chat_with_json.return_value = {
            "action": "switch_to",
            "paper_index": None,
            "message": "Switching..."
        }

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session.process_message("Go to paper...")

        assert "couldn't determine" in response.lower()

    def test_process_message_conversation_only(self, sample_search_results, mock_llm_client):
        """Test processing message with no action (conversation only)."""
        from magpie.core.interactive_review import InteractiveReviewSession

        mock_llm_client.chat_with_json.return_value = {
            "message": "I can help you with that question."
        }

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        response = session.process_message("What can you do?")

        assert response == "I can help you with that question."

    def test_build_review_prompt(self, sample_search_results, mock_llm_client):
        """Test building the review prompt."""
        from magpie.core.interactive_review import InteractiveReviewSession

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )

        prompt = session._build_review_prompt()

        assert "Paper One: Machine Learning" in prompt
        assert "1 of 3" in prompt
        assert "present_paper" in prompt
        assert "skip_current" in prompt

    def test_save_to_profile(self, sample_search_results, mock_llm_client):
        """Test saving session state to profile."""
        from magpie.core.interactive_review import InteractiveReviewSession
        from magpie.models.profile import UserProfile

        profile = UserProfile(user_id="test-user")

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )
        session.current_index = 1
        session.saved_to_zotero.add("arxiv:2301.00001")
        session.conversation_history.append({"role": "user", "content": "Hello"})
        session.last_response = "Welcome!"

        session.save_to_profile(profile)

        assert profile.active_review_session is not None
        assert profile.active_review_session["current_index"] == 1
        assert "arxiv:2301.00001" in profile.active_review_session["saved_to_zotero"]
        assert len(profile.active_review_session["conversation_history"]) == 1
        assert profile.active_review_session["last_response"] == "Welcome!"

    def test_restore_from_profile(self, sample_search_results, mock_llm_client):
        """Test restoring session from profile."""
        from magpie.core.interactive_review import InteractiveReviewSession
        from magpie.models.profile import UserProfile

        profile = UserProfile(user_id="test-user")

        # First save a session
        original_session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )
        original_session.current_index = 2
        original_session.saved_to_zotero.add("arxiv:2301.00002")
        original_session.conversation_history.append({"role": "user", "content": "Test"})
        original_session.last_response = "Response"
        original_session.save_to_profile(profile)

        # Restore from profile
        restored_session = InteractiveReviewSession.restore_from_profile(
            profile, mock_llm_client
        )

        assert restored_session is not None
        assert restored_session.current_index == 2
        assert "arxiv:2301.00002" in restored_session.saved_to_zotero
        assert len(restored_session.conversation_history) == 1
        assert restored_session.last_response == "Response"

    def test_restore_from_profile_no_session(self, mock_llm_client):
        """Test restoring when no session saved."""
        from magpie.core.interactive_review import InteractiveReviewSession
        from magpie.models.profile import UserProfile

        profile = UserProfile(user_id="test-user")

        restored = InteractiveReviewSession.restore_from_profile(profile, mock_llm_client)

        assert restored is None

    def test_clear_from_profile(self, sample_search_results, mock_llm_client):
        """Test clearing session from profile."""
        from magpie.core.interactive_review import InteractiveReviewSession
        from magpie.models.profile import UserProfile

        profile = UserProfile(user_id="test-user")

        session = InteractiveReviewSession(
            results=sample_search_results,
            llm_client=mock_llm_client
        )
        session.save_to_profile(profile)

        assert profile.active_review_session is not None

        session.clear_from_profile(profile)

        assert profile.active_review_session is None
