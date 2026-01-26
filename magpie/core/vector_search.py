"""
Vector search for Magpie.

Searches ChromaDB vector database for papers semantically similar to queries.
Handles multi-query search, deduplication, and result merging.
"""
import chromadb
import datetime
import typing

from magpie.core.query_processor import ProcessedQuery
from magpie.models.paper import Paper
from magpie.models.query import SubQuery
from magpie.models.results import SearchResults, PaperResult
from magpie.utils.config import Config


class VectorSearch:
    """
    Semantic search over paper database using vector embeddings.
    
    Searches ChromaDB for papers similar to query embeddings, handling
    multi-query searches with weighted result allocation and deduplication.
    """
    
    def __init__(
        self,
        db_path: typing.Optional[str] = None,
        collection_name: typing.Optional[str] = None
    ):
        """
        Initialize vector search.
        
        Args:
            db_path: Path to ChromaDB storage. If None, uses Config.VECTOR_DB_PATH
            collection_name: ChromaDB collection name. If None, uses Config.VECTOR_DB_COLLECTION
        """
        self.db_path = db_path or Config.VECTOR_DB_PATH
        self.collection_name = collection_name or Config.VECTOR_DB_COLLECTION
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Get collection (should already exist from indexing)
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception as e:
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                f"Run populate_papers.py first to create database. Error: {e}"
            )
    
    def search(self, processed_query: ProcessedQuery) -> SearchResults:
        """
        Search for papers matching the processed query.
        
        Executes independent searches for each subquery, allocates results
        proportionally by weight, merges and deduplicates results.
        
        Args:
            processed_query: ProcessedQuery with embeddings for each subquery
            
        Returns:
            SearchResults with deduplicated papers and relevance scores
        """
        query = processed_query.query
        embeddings = processed_query.embeddings
        
        # Calculate how many results each subquery should retrieve
        results_per_query = self._allocate_results(query.max_results, query.queries)
        
        # Execute searches for each subquery
        all_results: typing.Dict[str, typing.Dict] = {}  # paper_id_str -> result info
        
        for i, (subquery, embedding) in enumerate(zip(query.queries, embeddings)):
            n_results = results_per_query[i]
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=n_results,
                # TODO: Add filtering based on query.date_range, query.min_citations, etc.
            )
            
            # Process results from this subquery
            self._process_subquery_results(
                results=results,
                subquery=subquery,
                all_results=all_results
            )
        
        # Convert to PaperResult objects with boosted scores
        paper_results = self._build_paper_results(all_results)
        
        # Sort by relevance score (highest first)
        paper_results.sort(key=lambda pr: pr.relevance_score, reverse=True)
        
        # Trim to max_results (may have more due to overlap)
        paper_results = paper_results[:query.max_results]
        
        # Build SearchResults
        search_results = SearchResults(
            results=paper_results,
            query=query,
            total_found=len(all_results),  # Before trimming
            timestamp=datetime.datetime.now()
        )
        
        return search_results
    
    def _allocate_results(
        self,
        max_results: int,
        subqueries: typing.List[SubQuery]
    ) -> typing.List[int]:
        """
        Allocate number of results each subquery should retrieve based on weights.
        
        Args:
            max_results: Total maximum results desired
            subqueries: List of subqueries with weights
            
        Returns:
            List of result counts per subquery (same order as subqueries)
        """
        # Allocate proportionally to weights
        # Ensure each gets at least 1 result
        allocations = []
        for subquery in subqueries:
            n = max(1, int(max_results * subquery.weight))
            allocations.append(n)
        
        return allocations
    
    def _process_subquery_results(
        self,
        results: typing.Dict,
        subquery: SubQuery,
        all_results: typing.Dict[str, typing.Dict]
    ) -> None:
        """
        Process results from a single subquery search and add to all_results.
        
        Handles deduplication - if paper already in all_results, adds this
        subquery to its matched_subqueries list.
        
        Args:
            results: ChromaDB query results
            subquery: The subquery that generated these results
            all_results: Accumulated results dict (modified in place)
        """
        if not results["ids"] or not results["ids"][0]:
            return
        
        # ChromaDB returns nested lists: [[id1, id2, ...]]
        ids = results["ids"][0]
        distances = results["distances"][0]  # Cosine distance (lower = more similar)
        metadatas = results["metadatas"][0]
        
        for paper_id_str, distance, metadata in zip(ids, distances, metadatas):
            # Convert distance to similarity score (1 - distance for cosine)
            # ChromaDB uses L2 distance by default, but for cosine embeddings
            # the distance is actually cosine distance
            similarity_score = 1.0 - distance
            
            if paper_id_str in all_results:
                # Paper already found by another subquery - add this subquery
                all_results[paper_id_str]["matched_subqueries"].append(subquery)
                all_results[paper_id_str]["scores"].append(similarity_score)
            else:
                # New paper
                all_results[paper_id_str] = {
                    "metadata": metadata,
                    "matched_subqueries": [subquery],
                    "scores": [similarity_score]
                }
    
    def _build_paper_results(
        self,
        all_results: typing.Dict[str, typing.Dict]
    ) -> typing.List[PaperResult]:
        """
        Convert accumulated results to PaperResult objects with boosted scores.
        
        Papers matched by multiple subqueries get score boost.
        
        Args:
            all_results: Dict of paper_id_str -> result info
            
        Returns:
            List of PaperResult objects
        """
        paper_results = []
        
        for paper_id_str, info in all_results.items():
            # Reconstruct Paper from metadata
            paper = self._metadata_to_paper(info["metadata"])
            
            # Calculate boosted score
            # FIXME: Make boost multiplier configurable?
            base_score = max(info["scores"])  # Take best score
            num_queries_matched = len(info["matched_subqueries"])
            
            if num_queries_matched == 1:
                boosted_score = base_score
            elif num_queries_matched == 2:
                boosted_score = base_score * 1.5
            else:  # 3+
                boosted_score = base_score * 2.0
            
            # Clamp to [0, 1] range
            boosted_score = min(1.0, boosted_score)
            
            # Create PaperResult
            paper_result = PaperResult(
                paper=paper,
                matched_subqueries=info["matched_subqueries"],
                relevance_score=boosted_score,
                explanation=None  # Will be filled by synthesizer
            )
            
            paper_results.append(paper_result)
        
        return paper_results
    
    def _metadata_to_paper(self, metadata: typing.Dict) -> Paper:
        """
        Reconstruct Paper object from ChromaDB metadata.
        
        Args:
            metadata: ChromaDB metadata dict with paper_json field
            
        Returns:
            Paper object
        """
        paper_json = metadata["paper_json"]
        return Paper.model_validate_json(paper_json)
    
    def get_paper_count(self) -> int:
        """Get total number of papers in the database."""
        return self.collection.count()
