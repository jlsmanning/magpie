"""
Text embedding using Sentence-BERT.

Provides embedding functionality for converting text to dense vectors
for semantic search.
"""
#FIXME: if preprocessing gets complex avoid duplicate logic in embed_text() and embed_batch()
import typing
import numpy
import sentence_transformers

from magpie.utils.config import Config


class Embedder:
    """
    Wrapper for Sentence-BERT embedding models.
    
    Loads model once at initialization and provides methods for
    embedding single texts or batches.
    """
    
    def __init__(self, model_name: typing.Optional[str] = None):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Sentence-BERT model name. If None, uses Config.EMBEDDER_MODEL
        """
        self.model_name = model_name or Config.EMBEDDER_MODEL
        self.embedding_dim = Config.get_embedding_dim(self.model_name)
        
        # Load model (this is slow, so we do it once)
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
    
    def embed_text(self, text: str) -> numpy.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array of shape (embedding_dim,)
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # encode() returns array of shape (1, embedding_dim), we want just (embedding_dim,)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: typing.List[str]) -> numpy.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding matrix as numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")
        
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text in batch")
        
        # encode() with list returns array of shape (num_texts, embedding_dim)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim
    
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model_name
