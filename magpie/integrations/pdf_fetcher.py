"""
PDF fetching and text extraction for Magpie.

Downloads PDFs from URLs and extracts text content for analysis.
"""

import os
import pymupdf  # PyMuPDF for PDF text extraction
import requests
import typing

from magpie.utils.config import Config


class PDFFetcher:
    """
    Fetches and extracts text from PDF files.
    
    Downloads PDFs from URLs, caches them locally, and extracts text content.
    """
    
    def __init__(self, cache_dir: typing.Optional[str] = None):
        """
        Initialize PDF fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded PDFs. If None, uses ./data/pdfs/
        """
        self.cache_dir = cache_dir or "./data/pdfs"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def fetch_pdf(self, url: str, paper_id: str) -> str:
        """
        Download PDF from URL and return local file path.
        
        Uses cached version if already downloaded.
        
        Args:
            url: URL to PDF file
            paper_id: Unique identifier for paper (used for cache filename)
            
        Returns:
            Path to downloaded PDF file
            
        Raises:
            Exception: If download fails
            
        Example:
            >>> fetcher = PDFFetcher()
            >>> path = fetcher.fetch_pdf("https://arxiv.org/pdf/2301.12345.pdf", "arxiv:2301.12345")
        """
        # Generate cache filename
        safe_id = paper_id.replace(":", "_").replace("/", "_")
        cache_path = os.path.join(self.cache_dir, f"{safe_id}.pdf")
        
        # Return cached version if exists
        if os.path.exists(cache_path):
            return cache_path
        
        # Download PDF
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            return cache_path
            
        except Exception as e:
            raise Exception(f"Failed to download PDF from {url}: {e}")
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            Exception: If text extraction fails
        """
        try:
            doc = pymupdf.open(pdf_path)
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            
            doc.close()
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"Failed to extract text from {pdf_path}: {e}")
    
    def fetch_and_extract(self, url: str, paper_id: str) -> typing.Tuple[str, str]:
        """
        Download PDF and extract text in one call.
        
        Args:
            url: URL to PDF file
            paper_id: Unique identifier for paper
            
        Returns:
            Tuple of (pdf_path, extracted_text)
            
        Example:
            >>> fetcher = PDFFetcher()
            >>> path, text = fetcher.fetch_and_extract(url, paper_id)
            >>> print(f"Downloaded to {path}, extracted {len(text)} characters")
        """
        pdf_path = self.fetch_pdf(url, paper_id)
        text = self.extract_text(pdf_path)
        return pdf_path, text
    
    def extract_section(
        self,
        pdf_path: str,
        section_name: str,
        full_text: typing.Optional[str] = None
    ) -> typing.Optional[str]:
        """
        Extract a specific section from the PDF text.
        
        Looks for common section headers like "Methods", "Results", "Introduction", etc.
        This is a heuristic approach and may not work for all papers.
        
        Args:
            pdf_path: Path to PDF file
            section_name: Name of section to extract (case-insensitive)
            full_text: Pre-extracted full text (optional, will extract if not provided)
            
        Returns:
            Section text if found, None otherwise
        """
        if full_text is None:
            full_text = self.extract_text(pdf_path)
        
        # Common section header patterns
        section_patterns = [
            f"\n{section_name}\n",
            f"\n{section_name.upper()}\n",
            f"\n{section_name.lower()}\n",
            f"\n{section_name.capitalize()}\n",
        ]
        
        # Try to find section
        text_lower = full_text.lower()
        section_lower = section_name.lower()
        
        # Find section start
        section_start = -1
        for pattern in section_patterns:
            idx = full_text.find(pattern)
            if idx != -1:
                section_start = idx
                break
        
        if section_start == -1:
            # Section not found
            return None
        
        # Find next section (common headers)
        next_sections = [
            "introduction", "background", "related work",
            "methods", "methodology", "approach",
            "results", "experiments", "evaluation",
            "discussion", "conclusion", "references"
        ]
        
        # Find earliest next section
        section_end = len(full_text)
        for next_section in next_sections:
            if next_section == section_lower:
                continue  # Skip current section
            
            for pattern in [f"\n{next_section}\n", f"\n{next_section.upper()}\n"]:
                idx = full_text.find(pattern, section_start + 10)
                if idx != -1 and idx < section_end:
                    section_end = idx
        
        # Extract section
        section_text = full_text[section_start:section_end].strip()
        return section_text
    
    def clear_cache(self) -> int:
        """
        Delete all cached PDFs.
        
        Returns:
            Number of files deleted
        """
        count = 0
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pdf'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
        return count
