"""
Configuration for Magpie.

Centralized settings that can be overridden via environment variables or config file.
TODO: Add YAML config file support for easier configuration management.
"""
#FIXME: There are some ugly things here like hard coded LLM stuff, mix of user and implementation variables, and model dims should be inferred
import os
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()


class Config:
    """
    Configuration settings for Magpie.
    
    Settings can be overridden via environment variables with MAGPIE_ prefix.
    Example: MAGPIE_EMBEDDER_MODEL=all-MiniLM-L6-v2
    """
    
    # Embedder settings
    EMBEDDER_MODEL = os.getenv(
        "MAGPIE_EMBEDDER_MODEL",
        "all-mpnet-base-v2"  # High quality, 768 dimensions
    )
    EMBEDDING_DIM = 768  # Dimension for all-mpnet-base-v2
    
    # Vector database settings
    VECTOR_DB_PATH = os.getenv(
        "MAGPIE_VECTOR_DB_PATH",
        "./data/vector_db"
    )
    VECTOR_DB_COLLECTION = "papers"
    
    # LLM settings
    LLM_PROVIDER = os.getenv("MAGPIE_LLM_PROVIDER", "anthropic")  # or "openai"
    LLM_MODEL = os.getenv("MAGPIE_LLM_MODEL", "claude-sonnet-4-5-20250929")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Profile storage
    PROFILE_DIR = os.getenv("MAGPIE_PROFILE_DIR", "./data/profiles")
    DEFAULT_USER_ID = "default_user"
    
    # Search defaults
    DEFAULT_MAX_RESULTS = 10
    DEFAULT_RECENCY_WEIGHT = 0.5
    DEFAULT_MIN_CITATIONS = None
    SEEN_PAPERS_RETENTION_DAYS = 90
    
    # Paper sources
    ARXIV_MAX_RESULTS_PER_QUERY = 100
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    @classmethod
    def get_embedding_dim(cls, model_name: str) -> int:
        """Get embedding dimension for a given model."""
        # Map of known models to their dimensions
        dims = {
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
        }
        return dims.get(model_name, 768)  # Default to 768
