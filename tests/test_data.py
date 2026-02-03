"""
Tests for Magpie data components.

Tests cover arxiv_puller.py and paper_indexer.py.
"""

import datetime
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from magpie.models.paper import Paper


# =============================================================================
# ArXiv Puller Tests
# =============================================================================

class TestArxivPuller:
    """Tests for ArXiv paper fetching functions."""

    @pytest.fixture
    def mock_arxiv_result(self):
        """Create a mock ArXiv result object."""
        result = Mock()
        result.entry_id = "http://arxiv.org/abs/2301.12345v1"
        result.title = "A Great Paper on Machine Learning"
        result.summary = "This paper presents novel machine learning techniques."
        result.published = datetime.datetime(2024, 1, 15, 12, 0, 0)
        result.updated = datetime.datetime(2024, 1, 20, 12, 0, 0)
        result.pdf_url = "https://arxiv.org/pdf/2301.12345.pdf"
        result.doi = None
        result.comment = "Accepted at CVPR 2024"
        result.journal_ref = None
        result.primary_category = "cs.LG"
        result.categories = ["cs.LG", "cs.AI"]

        # Mock authors
        author1 = Mock()
        author1.name = "John Smith"
        author2 = Mock()
        author2.name = "Jane Doe"
        result.authors = [author1, author2]

        return result

    def test_arxiv_result_to_paper(self, mock_arxiv_result):
        """Test converting ArXiv result to Paper."""
        from magpie.data.arxiv_puller import _arxiv_result_to_paper

        paper = _arxiv_result_to_paper(mock_arxiv_result)

        assert paper.paper_id == ("arxiv", "2301.12345")
        assert paper.title == "A Great Paper on Machine Learning"
        assert paper.authors == ["John Smith", "Jane Doe"]
        assert paper.abstract == "This paper presents novel machine learning techniques."
        assert paper.published_date == datetime.date(2024, 1, 15)
        assert paper.categories == ["cs.LG", "cs.AI"]

    def test_arxiv_result_to_paper_extracts_venue(self, mock_arxiv_result):
        """Test that venue is extracted from comment."""
        from magpie.data.arxiv_puller import _arxiv_result_to_paper

        paper = _arxiv_result_to_paper(mock_arxiv_result)

        assert paper.venue == "CVPR 2024"

    def test_arxiv_result_to_paper_no_version(self, mock_arxiv_result):
        """Test ArXiv ID extraction without version."""
        from magpie.data.arxiv_puller import _arxiv_result_to_paper

        mock_arxiv_result.entry_id = "http://arxiv.org/abs/2301.12345"
        paper = _arxiv_result_to_paper(mock_arxiv_result)

        assert paper.paper_id == ("arxiv", "2301.12345")

    def test_extract_venue_from_comment(self):
        """Test venue extraction from various comment formats."""
        from magpie.data.arxiv_puller import _extract_venue_from_comment

        # Standard formats
        assert _extract_venue_from_comment("Accepted at CVPR 2024") == "CVPR 2024"
        assert _extract_venue_from_comment("Published in NeurIPS 2023") == "NEURIPS 2023"
        assert _extract_venue_from_comment("To appear at ICCV 2024") == "ICCV 2024"

        # Case insensitive
        assert _extract_venue_from_comment("accepted at cvpr 2024") == "CVPR 2024"

        # No venue
        assert _extract_venue_from_comment("10 pages, 5 figures") is None
        assert _extract_venue_from_comment(None) is None
        assert _extract_venue_from_comment("") is None

    def test_extract_venue_various_conferences(self):
        """Test extraction of various conference names."""
        from magpie.data.arxiv_puller import _extract_venue_from_comment

        venues = {
            "ICML 2024 paper": "ICML 2024",
            "Accepted to ICLR 2024": "ICLR 2024",
            "AAAI 2024": "AAAI 2024",
            "ACL 2023 findings": "ACL 2023",
            "EMNLP 2024 main conference": "EMNLP 2024",
        }

        for comment, expected in venues.items():
            result = _extract_venue_from_comment(comment)
            assert result == expected, f"Failed for '{comment}': got {result}"

    @patch('magpie.data.arxiv_puller.arxiv')
    def test_fetch_papers(self, mock_arxiv, mock_arxiv_result):
        """Test fetching papers from ArXiv."""
        from magpie.data.arxiv_puller import fetch_papers

        # Mock search results
        mock_search = Mock()
        mock_search.results.return_value = [mock_arxiv_result]
        mock_arxiv.Search.return_value = mock_search
        mock_arxiv.SortCriterion.SubmittedDate = "submitted"
        mock_arxiv.SortOrder.Descending = "desc"

        papers = fetch_papers(
            categories=["cs.LG"],
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 12, 31),
            max_results=10
        )

        assert len(papers) == 1
        assert papers[0].title == "A Great Paper on Machine Learning"

    @patch('magpie.data.arxiv_puller.arxiv')
    def test_fetch_papers_filters_by_date(self, mock_arxiv, mock_arxiv_result):
        """Test that fetch_papers filters by date range."""
        from magpie.data.arxiv_puller import fetch_papers

        # Paper is from Jan 15, 2024
        mock_search = Mock()
        mock_search.results.return_value = [mock_arxiv_result]
        mock_arxiv.Search.return_value = mock_search
        mock_arxiv.SortCriterion.SubmittedDate = "submitted"
        mock_arxiv.SortOrder.Descending = "desc"

        # Date range that excludes the paper
        papers = fetch_papers(
            categories=["cs.LG"],
            start_date=datetime.date(2024, 2, 1),  # After paper date
            end_date=datetime.date(2024, 12, 31),
            max_results=10
        )

        assert len(papers) == 0

    @patch('magpie.data.arxiv_puller.arxiv')
    def test_fetch_papers_with_keywords(self, mock_arxiv, mock_arxiv_result):
        """Test fetching papers with keyword filter."""
        from magpie.data.arxiv_puller import fetch_papers

        mock_search = Mock()
        mock_search.results.return_value = [mock_arxiv_result]
        mock_arxiv.Search.return_value = mock_search
        mock_arxiv.SortCriterion.SubmittedDate = "submitted"
        mock_arxiv.SortOrder.Descending = "desc"

        papers = fetch_papers(
            categories=["cs.LG"],
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 12, 31),
            max_results=10,
            keywords=["machine learning", "neural networks"]
        )

        # Check that query includes keywords
        call_args = mock_arxiv.Search.call_args
        query = call_args.kwargs['query']
        assert "all:machine learning" in query or "all:neural networks" in query

    @patch('magpie.data.arxiv_puller.arxiv')
    def test_fetch_recent_papers(self, mock_arxiv, mock_arxiv_result):
        """Test fetch_recent_papers convenience function."""
        from magpie.data.arxiv_puller import fetch_recent_papers

        # Make paper date recent
        mock_arxiv_result.published = datetime.datetime.now()

        mock_search = Mock()
        mock_search.results.return_value = [mock_arxiv_result]
        mock_arxiv.Search.return_value = mock_search
        mock_arxiv.SortCriterion.SubmittedDate = "submitted"
        mock_arxiv.SortOrder.Descending = "desc"

        papers = fetch_recent_papers(
            categories=["cs.LG"],
            days_back=7,
            max_results=50
        )

        assert len(papers) == 1


# =============================================================================
# Paper Indexer Tests
# =============================================================================

class TestPaperIndexer:
    """Tests for PaperIndexer class."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create temporary database directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        return [
            Paper(
                paper_id=("arxiv", "2301.00001"),
                title="Paper One on Machine Learning",
                authors=["Author A"],
                abstract="This paper explores machine learning.",
                published_date=datetime.date(2024, 1, 15),
                url="https://arxiv.org/abs/2301.00001",
                categories=["cs.LG"]
            ),
            Paper(
                paper_id=("arxiv", "2301.00002"),
                title="Paper Two on Computer Vision",
                authors=["Author B"],
                abstract="This paper explores computer vision.",
                published_date=datetime.date(2024, 2, 20),
                url="https://arxiv.org/abs/2301.00002",
                categories=["cs.CV"]
            ),
        ]

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        mock = Mock()
        mock.embed_batch = Mock(return_value=np.random.rand(2, 768))
        return mock

    def test_indexer_init(self, temp_db_dir, mock_embedder):
        """Test indexer initialization."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        assert indexer.db_path == temp_db_dir
        assert indexer.collection_name == "test_papers"
        assert indexer.collection is not None

    def test_index_papers(self, temp_db_dir, sample_papers, mock_embedder):
        """Test indexing papers."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        result = indexer.index_papers(sample_papers)

        assert result["total"] == 2
        assert result["indexed"] == 2
        assert result["skipped"] == 0
        assert result["failed"] == 0
        assert indexer.get_paper_count() == 2

    def test_index_papers_skip_existing(self, temp_db_dir, sample_papers, mock_embedder):
        """Test that existing papers are skipped."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        # Index once
        indexer.index_papers(sample_papers)

        # Reset mock for second call
        mock_embedder.embed_batch.reset_mock()

        # Index again
        result = indexer.index_papers(sample_papers, skip_existing=True)

        assert result["indexed"] == 0
        assert result["skipped"] == 2
        mock_embedder.embed_batch.assert_not_called()

    def test_index_papers_empty_list(self, temp_db_dir, mock_embedder):
        """Test indexing empty list."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        result = indexer.index_papers([])

        assert result == {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}

    def test_paper_exists(self, temp_db_dir, sample_papers, mock_embedder):
        """Test checking if paper exists."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        indexer.index_papers(sample_papers)

        assert indexer.paper_exists(("arxiv", "2301.00001")) is True
        assert indexer.paper_exists(("arxiv", "9999.99999")) is False

    def test_delete_paper(self, temp_db_dir, sample_papers, mock_embedder):
        """Test deleting a paper."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        indexer.index_papers(sample_papers)
        assert indexer.get_paper_count() == 2

        result = indexer.delete_paper(("arxiv", "2301.00001"))

        assert result is True
        assert indexer.get_paper_count() == 1
        assert indexer.paper_exists(("arxiv", "2301.00001")) is False

    def test_delete_paper_not_found(self, temp_db_dir, mock_embedder):
        """Test deleting paper that doesn't exist."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        result = indexer.delete_paper(("arxiv", "9999.99999"))

        # ChromaDB delete doesn't error on missing IDs
        assert result is True

    def test_paper_id_to_str(self, temp_db_dir, mock_embedder):
        """Test paper ID to string conversion."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        assert indexer._paper_id_to_str(("arxiv", "2301.12345")) == "arxiv:2301.12345"
        assert indexer._paper_id_to_str(("doi", "10.1234/test")) == "doi:10.1234/test"

    def test_str_to_paper_id(self, temp_db_dir, mock_embedder):
        """Test string to paper ID conversion."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        assert indexer._str_to_paper_id("arxiv:2301.12345") == ("arxiv", "2301.12345")
        assert indexer._str_to_paper_id("doi:10.1234/test") == ("doi", "10.1234/test")

    def test_paper_to_metadata(self, temp_db_dir, sample_papers, mock_embedder):
        """Test converting paper to metadata."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        metadata = indexer._paper_to_metadata(sample_papers[0])

        assert "paper_json" in metadata
        assert metadata["source"] == "arxiv"
        assert metadata["paper_id"] == "2301.00001"
        assert metadata["published_year"] == 2024
        assert metadata["title"] == "Paper One on Machine Learning"
        assert metadata["primary_category"] == "cs.LG"

    def test_get_paper_count(self, temp_db_dir, sample_papers, mock_embedder):
        """Test getting paper count."""
        from magpie.data.paper_indexer import PaperIndexer

        indexer = PaperIndexer(
            db_path=temp_db_dir,
            embedder=mock_embedder,
            collection_name="test_papers"
        )

        assert indexer.get_paper_count() == 0

        indexer.index_papers(sample_papers)

        assert indexer.get_paper_count() == 2
