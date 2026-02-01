"""
Zotero integration for Magpie.

Saves research papers to Zotero library with metadata and optional PDF attachments.
"""

from pyzotero import zotero
import typing

from magpie.models.paper import Paper
from magpie.utils.config import Config


class ZoteroClient:
    """
    Client for interacting with Zotero library.
    
    Handles saving papers to a designated collection, creating the collection
    if it doesn't exist, and attaching PDFs.
    """
    
    def __init__(
        self,
        library_id: typing.Optional[str] = None,
        library_type: typing.Optional[str] = None,
        api_key: typing.Optional[str] = None,
        collection_name: typing.Optional[str] = None
    ):
        """
        Initialize Zotero client.
        
        Args:
            library_id: Zotero library ID. If None, uses Config.ZOTERO_LIBRARY_ID
            library_type: "user" or "group". If None, uses Config.ZOTERO_LIBRARY_TYPE
            api_key: Zotero API key. If None, uses Config.ZOTERO_API_KEY
            collection_name: Collection to save papers to. If None, uses Config.ZOTERO_COLLECTION_NAME
            
        Raises:
            ValueError: If credentials not provided and not in config
        """
        self.library_id = library_id or Config.ZOTERO_LIBRARY_ID
        self.library_type = library_type or Config.ZOTERO_LIBRARY_TYPE
        self.api_key = api_key or Config.ZOTERO_API_KEY
        self.collection_name = collection_name or Config.ZOTERO_COLLECTION_NAME
        
        if not self.library_id or not self.api_key:
            raise ValueError(
                "Zotero credentials not configured. "
                "Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY in .env file."
            )
        
        # Initialize pyzotero client
        self.zot = zotero.Zotero(
            library_id=self.library_id,
            library_type=self.library_type,
            api_key=self.api_key
        )
        
        # Get or create collection
        self.collection_key = self._get_or_create_collection()
    
    def save_paper(
        self,
        paper: Paper,
        tags: typing.Optional[typing.List[str]] = None,
        attach_pdf: bool = False,
        pdf_path: typing.Optional[str] = None
    ) -> str:
        """
        Save paper to Zotero library.
        
        Args:
            paper: Paper object to save
            tags: Optional list of tags to add to the item
            attach_pdf: If True, attempt to attach PDF (requires pdf_path or paper.pdf_url)
            pdf_path: Path to PDF file. If None and attach_pdf=True, downloads from paper.pdf_url
            
        Returns:
            Zotero item key of created item
            
        Example:
            >>> client = ZoteroClient()
            >>> item_key = client.save_paper(paper, tags=["fairness", "computer vision"])
        """
        # Convert Paper to Zotero item format
        item_data = self._paper_to_zotero_item(paper, tags)
        
        # Create item in Zotero
        response = self.zot.create_items([item_data])
        
        # Extract item key from response
        if response['successful']:
            item_key = response['successful']['0']['key']
            
            # Add to collection
            self.zot.addto_collection(self.collection_key, item_key)
            
            # Attach PDF if requested
            if attach_pdf:
                if pdf_path:
                    self._attach_pdf_file(item_key, pdf_path)
                elif paper.pdf_url:
                    # TODO: Download PDF and attach
                    # Would need PDF fetcher integration
                    print(f"Note: PDF attachment from URL not yet implemented. PDF available at: {paper.pdf_url}")
            
            return item_key
        else:
            raise Exception(f"Failed to create Zotero item: {response}")
    
    def _get_or_create_collection(self) -> str:
        """
        Get collection key for designated collection, creating if doesn't exist.
        
        Returns:
            Collection key (Zotero's internal ID for the collection)
        """
        # Get all collections
        collections = self.zot.collections()
        
        # Look for our collection
        for col in collections:
            if col['data']['name'] == self.collection_name:
                return col['key']
        
        # Collection doesn't exist - create it
        new_collection_data = {
            'name': self.collection_name,
            'parentCollection': False
        }
        
        response = self.zot.create_collections([new_collection_data])
        
        if response['successful']:
            collection_key = response['successful']['0']['key']
            print(f"Created Zotero collection: {self.collection_name}")
            return collection_key
        else:
            raise Exception(f"Failed to create Zotero collection: {response}")
    
    def _paper_to_zotero_item(
        self,
        paper: Paper,
        tags: typing.Optional[typing.List[str]] = None
    ) -> typing.Dict[str, typing.Any]:
        """
        Convert Paper object to Zotero item format.
        
        Args:
            paper: Paper to convert
            tags: Optional tags to add
            
        Returns:
            Dictionary in Zotero item format
        """
        # Determine item type based on paper source
        # ArXiv papers are preprints
        if paper.paper_id[0] == "arxiv":
            item_type = "preprint"
        else:
            item_type = "journalArticle"
        
        # Convert authors to Zotero creator format
        creators = []
        for author in paper.authors:
            # Try to split into first/last name
            # This is heuristic - may not work for all name formats
            parts = author.rsplit(' ', 1)
            if len(parts) == 2:
                creators.append({
                    'creatorType': 'author',
                    'firstName': parts[0],
                    'lastName': parts[1]
                })
            else:
                creators.append({
                    'creatorType': 'author',
                    'name': author  # Use full name if can't split
                })
        
        # Build Zotero item
        item = {
            'itemType': item_type,
            'title': paper.title,
            'creators': creators,
            'abstractNote': paper.abstract,
            'date': str(paper.published_date),
            'url': paper.url,
        }
        
        # Add optional fields
        if paper.doi:
            item['DOI'] = paper.doi
        
        if paper.venue:
            item['publicationTitle'] = paper.venue
        
        # Add tags
        if tags:
            item['tags'] = [{'tag': tag} for tag in tags]
        
        # Add ArXiv ID as extra field for preprints
        if paper.paper_id[0] == "arxiv":
            item['archiveID'] = paper.paper_id[1]
            item['repository'] = "arXiv"
        
        return item
    
    def _attach_pdf_file(self, item_key: str, pdf_path: str) -> None:
        """
        Attach PDF file to Zotero item.
        
        Args:
            item_key: Zotero item key to attach to
            pdf_path: Path to PDF file on disk
        """
        try:
            self.zot.attachment_simple([pdf_path], item_key)
        except Exception as e:
            print(f"Warning: Failed to attach PDF: {e}")
    
    def search_items(self, query: str) -> typing.List[typing.Dict]:
        """
        Search for items in library.
        
        Args:
            query: Search query
            
        Returns:
            List of matching items
        """
        return self.zot.items(q=query)
    
    def get_collections(self) -> typing.List[typing.Dict]:
        """
        Get all collections in library.
        
        Returns:
            List of collection objects
        """
        return self.zot.collections()
