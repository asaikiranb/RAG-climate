"""
PDF text extraction using PyMuPDF.
Extracts text with page numbers and metadata.
"""

from typing import List, Dict
from pathlib import Path
import fitz  # PyMuPDF

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import IngestionError

logger = get_logger(__name__)


class PDFExtractor:
    """Extract text from PDF files with metadata."""

    def __init__(self, settings: Settings = None):
        """
        Initialize PDF extractor.

        Args:
            settings: Application settings (optional)
        """
        self.settings = settings

    def extract(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page-level granularity.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dicts with text and metadata per page
        """
        if not pdf_path.exists():
            raise IngestionError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting text from: {pdf_path.name}")

        try:
            doc = fitz.open(pdf_path)
            pages_data = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Only include pages with actual text content
                if text.strip():
                    pages_data.append({
                        'text': text,
                        'page_number': page_num + 1,  # 1-indexed
                        'filename': pdf_path.name,
                        'filepath': str(pdf_path)
                    })

            doc.close()

            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            return pages_data

        except Exception as e:
            raise IngestionError(f"Failed to extract PDF {pdf_path.name}: {e}")

    def extract_with_images(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text AND images from PDF.
        (Advanced feature for future multi-modal support)

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dicts with text, images, and metadata
        """
        # TODO: Implement image extraction for multi-modal RAG
        raise NotImplementedError("Image extraction not yet implemented")
