"""
Tests for Magpie integrations.

Tests cover embedder.py, llm_client.py, pdf_fetcher.py, and zotero_client.py.
Many tests use mocking since these integrate with external services.
"""

import datetime
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from magpie.models.paper import Paper


# =============================================================================
# Embedder Tests
# =============================================================================

class TestEmbedder:
    """Tests for Embedder class."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance (loads model - slow)."""
        from magpie.integrations.embedder import Embedder
        return Embedder()

    def test_embedder_init(self, embedder):
        """Test embedder initialization."""
        assert embedder.model is not None
        assert embedder.model_name == "all-mpnet-base-v2"
        assert embedder.embedding_dim == 768

    def test_embed_text(self, embedder):
        """Test embedding single text."""
        embedding = embedder.embed_text("machine learning")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_embed_text_empty_raises(self, embedder):
        """Test that empty text raises error."""
        with pytest.raises(ValueError) as exc_info:
            embedder.embed_text("")
        assert "empty" in str(exc_info.value).lower()

    def test_embed_text_whitespace_raises(self, embedder):
        """Test that whitespace-only text raises error."""
        with pytest.raises(ValueError) as exc_info:
            embedder.embed_text("   ")
        assert "empty" in str(exc_info.value).lower()

    def test_embed_batch(self, embedder):
        """Test embedding batch of texts."""
        texts = ["machine learning", "computer vision", "natural language processing"]
        embeddings = embedder.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)

    def test_embed_batch_empty_raises(self, embedder):
        """Test that empty batch raises error."""
        with pytest.raises(ValueError) as exc_info:
            embedder.embed_batch([])
        assert "empty" in str(exc_info.value).lower()

    def test_embed_batch_with_empty_text_raises(self, embedder):
        """Test that batch with empty text raises error."""
        with pytest.raises(ValueError) as exc_info:
            embedder.embed_batch(["valid text", "", "another text"])
        assert "empty" in str(exc_info.value).lower()

    def test_get_embedding_dim(self, embedder):
        """Test getting embedding dimension."""
        assert embedder.get_embedding_dim() == 768

    def test_get_model_name(self, embedder):
        """Test getting model name."""
        assert embedder.get_model_name() == "all-mpnet-base-v2"

    def test_embeddings_are_normalized(self, embedder):
        """Test that embeddings are roughly unit length (for cosine similarity)."""
        embedding = embedder.embed_text("test text")
        norm = np.linalg.norm(embedding)
        # Sentence transformers typically normalize, should be close to 1
        assert 0.9 < norm < 1.1


# =============================================================================
# ClaudeClient Tests (Mocked)
# =============================================================================

class TestClaudeClient:
    """Tests for ClaudeClient class (with mocked API)."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock the anthropic module."""
        with patch('magpie.integrations.llm_client.anthropic') as mock:
            # Mock the client
            mock_client = MagicMock()
            mock.Anthropic.return_value = mock_client

            # Mock a response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="This is the response.")]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_client.messages.create.return_value = mock_response

            mock.NOT_GIVEN = object()
            mock.APIError = Exception

            yield mock

    def test_client_init_with_key(self, mock_anthropic, monkeypatch):
        """Test client initialization with API key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from magpie.integrations.llm_client import ClaudeClient
        client = ClaudeClient(api_key="test-api-key")

        assert client.api_key == "test-api-key"
        mock_anthropic.Anthropic.assert_called_once()

    def test_client_init_no_key_raises(self, mock_anthropic, monkeypatch):
        """Test that missing API key raises error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Also patch Config to return None for API key
        from magpie.utils import config
        monkeypatch.setattr(config.Config, "ANTHROPIC_API_KEY", None)

        from magpie.integrations.llm_client import ClaudeClient

        with pytest.raises(ValueError) as exc_info:
            ClaudeClient(api_key=None)
        assert "API key" in str(exc_info.value)

    def test_chat(self, mock_anthropic, monkeypatch):
        """Test chat method."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from magpie.integrations.llm_client import ClaudeClient
        client = ClaudeClient(api_key="test-key")

        response = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            system="Be helpful."
        )

        assert response == "This is the response."
        assert client.total_input_tokens == 100
        assert client.total_output_tokens == 50

    def test_chat_with_json(self, mock_anthropic, monkeypatch):
        """Test chat_with_json method."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Mock JSON response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"key": "value", "number": 42}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

        from magpie.integrations.llm_client import ClaudeClient
        client = ClaudeClient(api_key="test-key")

        response = client.chat_with_json(
            messages=[{"role": "user", "content": "Return JSON"}]
        )

        assert response == {"key": "value", "number": 42}

    def test_chat_with_json_markdown_fences(self, mock_anthropic, monkeypatch):
        """Test chat_with_json handles markdown code fences."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Mock response with markdown fences
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"key": "value"}\n```')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

        from magpie.integrations.llm_client import ClaudeClient
        client = ClaudeClient(api_key="test-key")

        response = client.chat_with_json(
            messages=[{"role": "user", "content": "Return JSON"}]
        )

        assert response == {"key": "value"}

    def test_get_usage_stats(self, mock_anthropic, monkeypatch):
        """Test getting usage statistics."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from magpie.integrations.llm_client import ClaudeClient
        client = ClaudeClient(api_key="test-key")

        # Make a call to accumulate tokens
        client.chat(messages=[{"role": "user", "content": "Hi"}])

        stats = client.get_usage_stats()

        assert stats["input_tokens"] == 100
        assert stats["output_tokens"] == 50
        assert stats["total_tokens"] == 150

    def test_estimate_cost(self, mock_anthropic, monkeypatch):
        """Test cost estimation."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from magpie.integrations.llm_client import ClaudeClient
        client = ClaudeClient(api_key="test-key")

        # Set token counts directly for predictable test
        client.total_input_tokens = 1_000_000
        client.total_output_tokens = 1_000_000

        cost = client.estimate_cost()

        # $3/M input + $15/M output = $18
        assert cost == 18.0


# =============================================================================
# PDFFetcher Tests
# =============================================================================

class TestPDFFetcher:
    """Tests for PDFFetcher class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def pdf_fetcher(self, temp_cache_dir):
        """Create PDF fetcher with temp cache."""
        from magpie.integrations.pdf_fetcher import PDFFetcher
        return PDFFetcher(cache_dir=temp_cache_dir)

    def test_init_creates_cache_dir(self, temp_cache_dir):
        """Test that init creates cache directory."""
        import shutil
        shutil.rmtree(temp_cache_dir)  # Remove it first

        from magpie.integrations.pdf_fetcher import PDFFetcher
        fetcher = PDFFetcher(cache_dir=temp_cache_dir)

        assert os.path.exists(temp_cache_dir)

    @patch('magpie.integrations.pdf_fetcher.requests')
    def test_fetch_pdf(self, mock_requests, pdf_fetcher, temp_cache_dir):
        """Test fetching PDF from URL."""
        # Mock response
        mock_response = Mock()
        mock_response.content = b"PDF content here"
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        path = pdf_fetcher.fetch_pdf(
            "https://arxiv.org/pdf/2301.12345.pdf",
            "arxiv:2301.12345"
        )

        assert os.path.exists(path)
        assert path.endswith(".pdf")
        mock_requests.get.assert_called_once()

    @patch('magpie.integrations.pdf_fetcher.requests')
    def test_fetch_pdf_uses_cache(self, mock_requests, pdf_fetcher, temp_cache_dir):
        """Test that fetching uses cached PDF if available."""
        # Create a cached file
        cache_path = os.path.join(temp_cache_dir, "arxiv_2301.12345.pdf")
        with open(cache_path, 'wb') as f:
            f.write(b"cached PDF")

        path = pdf_fetcher.fetch_pdf(
            "https://arxiv.org/pdf/2301.12345.pdf",
            "arxiv:2301.12345"
        )

        assert path == cache_path
        mock_requests.get.assert_not_called()

    @patch('magpie.integrations.pdf_fetcher.requests')
    def test_fetch_pdf_error(self, mock_requests, pdf_fetcher):
        """Test fetch_pdf handles errors."""
        mock_requests.get.side_effect = Exception("Network error")

        with pytest.raises(Exception) as exc_info:
            pdf_fetcher.fetch_pdf("https://example.com/paper.pdf", "test:123")

        assert "Failed to download" in str(exc_info.value)

    @patch('magpie.integrations.pdf_fetcher.pymupdf')
    def test_extract_text(self, mock_pymupdf, pdf_fetcher, temp_cache_dir):
        """Test extracting text from PDF."""
        # Create mock PDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 text content."
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_pymupdf.open.return_value = mock_doc

        # Create a dummy PDF file
        pdf_path = os.path.join(temp_cache_dir, "test.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(b"dummy pdf")

        text = pdf_fetcher.extract_text(pdf_path)

        assert text == "Page 1 text content."

    @patch('magpie.integrations.pdf_fetcher.pymupdf')
    @patch('magpie.integrations.pdf_fetcher.requests')
    def test_fetch_and_extract(self, mock_requests, mock_pymupdf, pdf_fetcher):
        """Test combined fetch and extract."""
        # Mock download
        mock_response = Mock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        # Mock PDF extraction
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Extracted text."
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_pymupdf.open.return_value = mock_doc

        path, text = pdf_fetcher.fetch_and_extract(
            "https://arxiv.org/pdf/2301.12345.pdf",
            "arxiv:2301.12345"
        )

        assert path.endswith(".pdf")
        assert text == "Extracted text."

    def test_clear_cache(self, pdf_fetcher, temp_cache_dir):
        """Test clearing cache."""
        # Create some PDF files
        for i in range(3):
            path = os.path.join(temp_cache_dir, f"paper{i}.pdf")
            with open(path, 'wb') as f:
                f.write(b"content")

        count = pdf_fetcher.clear_cache()

        assert count == 3
        assert len(os.listdir(temp_cache_dir)) == 0

    def test_extract_section(self, pdf_fetcher, temp_cache_dir):
        """Test extracting specific section from text."""
        full_text = """
Introduction
This is the introduction section.

Methods
This is the methods section with details.

Results
These are the results.

Conclusion
This is the conclusion.
"""

        section = pdf_fetcher.extract_section(
            "/dummy/path.pdf",
            "Methods",
            full_text=full_text
        )

        assert section is not None
        assert "methods section" in section.lower()

    def test_extract_section_not_found(self, pdf_fetcher):
        """Test extracting section that doesn't exist."""
        full_text = "This is a paper with no clear sections."

        section = pdf_fetcher.extract_section(
            "/dummy/path.pdf",
            "Methods",
            full_text=full_text
        )

        assert section is None


# =============================================================================
# ZoteroClient Tests (Mocked)
# =============================================================================

class TestZoteroClient:
    """Tests for ZoteroClient class (with mocked API)."""

    @pytest.fixture
    def sample_paper(self):
        """Create sample paper for testing."""
        return Paper(
            paper_id=("arxiv", "2301.12345"),
            title="Test Paper Title",
            authors=["John Smith", "Jane Doe"],
            abstract="This is the abstract.",
            published_date=datetime.date(2024, 1, 15),
            url="https://arxiv.org/abs/2301.12345",
            doi="10.1234/test"
        )

    @patch('magpie.integrations.zotero_client.zotero')
    def test_client_init(self, mock_zotero, monkeypatch):
        """Test client initialization."""
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
        monkeypatch.setenv("ZOTERO_API_KEY", "test-key")

        # Mock Zotero client
        mock_zot = MagicMock()
        mock_zot.collections.return_value = []
        mock_zot.create_collections.return_value = {
            'successful': {'0': {'key': 'COL123'}}
        }
        mock_zotero.Zotero.return_value = mock_zot

        from magpie.integrations.zotero_client import ZoteroClient
        client = ZoteroClient(
            library_id="12345",
            api_key="test-key"
        )

        assert client.library_id == "12345"
        mock_zotero.Zotero.assert_called_once()

    def test_client_init_no_credentials_raises(self, monkeypatch):
        """Test that missing credentials raises error."""
        monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
        monkeypatch.delenv("ZOTERO_API_KEY", raising=False)

        # Also patch Config to return None for credentials
        from magpie.utils import config
        monkeypatch.setattr(config.Config, "ZOTERO_LIBRARY_ID", None)
        monkeypatch.setattr(config.Config, "ZOTERO_API_KEY", None)

        from magpie.integrations.zotero_client import ZoteroClient

        with pytest.raises(ValueError) as exc_info:
            ZoteroClient(library_id=None, api_key=None)
        assert "not configured" in str(exc_info.value)

    @patch('magpie.integrations.zotero_client.zotero')
    def test_save_paper(self, mock_zotero, sample_paper, monkeypatch):
        """Test saving paper to Zotero."""
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
        monkeypatch.setenv("ZOTERO_API_KEY", "test-key")

        # Mock Zotero client
        mock_zot = MagicMock()
        mock_zot.collections.return_value = [
            {'data': {'name': 'magpie'}, 'key': 'COL123'}
        ]
        mock_zot.create_items.return_value = {
            'successful': {'0': {'key': 'ITEM456'}}
        }
        mock_zotero.Zotero.return_value = mock_zot

        from magpie.integrations.zotero_client import ZoteroClient
        client = ZoteroClient(library_id="12345", api_key="test-key")

        item_key = client.save_paper(sample_paper, tags=["ML", "test"])

        assert item_key == "ITEM456"
        mock_zot.create_items.assert_called_once()
        mock_zot.addto_collection.assert_called_once_with("COL123", "ITEM456")

    @patch('magpie.integrations.zotero_client.zotero')
    def test_paper_to_zotero_item(self, mock_zotero, sample_paper, monkeypatch):
        """Test converting Paper to Zotero item format."""
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
        monkeypatch.setenv("ZOTERO_API_KEY", "test-key")

        mock_zot = MagicMock()
        mock_zot.collections.return_value = [
            {'data': {'name': 'magpie'}, 'key': 'COL123'}
        ]
        mock_zotero.Zotero.return_value = mock_zot

        from magpie.integrations.zotero_client import ZoteroClient
        client = ZoteroClient(library_id="12345", api_key="test-key")

        item = client._paper_to_zotero_item(sample_paper, tags=["tag1"])

        assert item['itemType'] == 'preprint'  # ArXiv papers are preprints
        assert item['title'] == "Test Paper Title"
        assert len(item['creators']) == 2
        assert item['DOI'] == "10.1234/test"
        assert item['tags'] == [{'tag': 'tag1'}]

    @patch('magpie.integrations.zotero_client.zotero')
    def test_get_or_create_collection_existing(self, mock_zotero, monkeypatch):
        """Test getting existing collection."""
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
        monkeypatch.setenv("ZOTERO_API_KEY", "test-key")

        mock_zot = MagicMock()
        mock_zot.collections.return_value = [
            {'data': {'name': 'magpie'}, 'key': 'EXISTING123'}
        ]
        mock_zotero.Zotero.return_value = mock_zot

        from magpie.integrations.zotero_client import ZoteroClient
        client = ZoteroClient(library_id="12345", api_key="test-key")

        assert client.collection_key == "EXISTING123"
        mock_zot.create_collections.assert_not_called()

    @patch('magpie.integrations.zotero_client.zotero')
    def test_get_or_create_collection_new(self, mock_zotero, monkeypatch):
        """Test creating new collection."""
        monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
        monkeypatch.setenv("ZOTERO_API_KEY", "test-key")

        mock_zot = MagicMock()
        mock_zot.collections.return_value = []  # No existing collections
        mock_zot.create_collections.return_value = {
            'successful': {'0': {'key': 'NEW456'}}
        }
        mock_zotero.Zotero.return_value = mock_zot

        from magpie.integrations.zotero_client import ZoteroClient
        client = ZoteroClient(library_id="12345", api_key="test-key")

        assert client.collection_key == "NEW456"
        mock_zot.create_collections.assert_called_once()
