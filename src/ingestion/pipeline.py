"""
Ingestion pipeline orchestrates the full document processing flow.
PDF → Text Extraction → Chunking → Embedding → Vector Store
"""

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

from src.config import Settings
from src.core.embeddings import EmbeddingModel
from src.core.vector_store import VectorStore
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.text_chunker import get_chunker
from src.utils.logger import get_logger, with_trace_id
from src.utils.exceptions import IngestionError

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(self, settings: Settings):
        """
        Initialize ingestion pipeline.

        Args:
            settings: Application settings
        """
        self.settings = settings

        logger.info("Initializing ingestion pipeline")

        # Initialize components
        self.pdf_extractor = PDFExtractor(settings)
        self.chunker = get_chunker(settings)
        self.embedding_model = EmbeddingModel(settings)
        self.vector_store = VectorStore(settings)

        logger.info("Ingestion pipeline ready")

    @with_trace_id
    def ingest_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """
        Ingest a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion for: {pdf_path.name}")

        try:
            # Step 1: Extract text from PDF
            logger.info("Step 1/4: Extracting text from PDF")
            pages = self.pdf_extractor.extract(pdf_path)

            if not pages:
                logger.warning(f"No text extracted from {pdf_path.name}")
                return {"status": "skipped", "reason": "no_text"}

            # Step 2: Chunk the text
            logger.info("Step 2/4: Chunking text")
            all_chunks = []

            for page in tqdm(pages, desc="Chunking pages"):
                page_metadata = {
                    'filename': page['filename'],
                    'page_number': str(page['page_number']),
                    'filepath': page['filepath']
                }

                chunks = self.chunker.chunk(page['text'], metadata=page_metadata)
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning(f"No chunks created from {pdf_path.name}")
                return {"status": "skipped", "reason": "no_chunks"}

            logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")

            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings")
            chunk_texts = [chunk['text'] for chunk in all_chunks]

            embeddings = self.embedding_model.encode_documents(
                chunk_texts,
                show_progress=True
            )

            # Step 4: Store in vector database
            logger.info("Step 4/4: Storing in vector database")
            documents = chunk_texts
            metadatas = [chunk['metadata'] for chunk in all_chunks]

            self.vector_store.add_documents_batch(
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                batch_size=100
            )

            stats = {
                "status": "success",
                "filename": pdf_path.name,
                "pages": len(pages),
                "chunks": len(all_chunks),
                "method": self.settings.chunking.method
            }

            logger.info(f"Successfully ingested {pdf_path.name}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path.name}: {e}")
            raise IngestionError(f"Ingestion failed for {pdf_path.name}: {e}")

    @with_trace_id
    def ingest_folder(self, folder_path: Path) -> Dict[str, any]:
        """
        Ingest all PDFs from a folder.

        Args:
            folder_path: Path to folder containing PDFs

        Returns:
            Dictionary with ingestion statistics
        """
        if not folder_path.exists():
            raise IngestionError(f"Folder not found: {folder_path}")

        # Find all PDF files
        pdf_files = list(folder_path.glob("*.pdf"))

        if not pdf_files:
            raise IngestionError(f"No PDF files found in {folder_path}")

        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")

        # Track stats
        total_stats = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "files": []
        }

        # Process each PDF
        for pdf_path in pdf_files:
            try:
                stats = self.ingest_pdf(pdf_path)

                if stats["status"] == "success":
                    total_stats["successful"] += 1
                    total_stats["total_pages"] += stats["pages"]
                    total_stats["total_chunks"] += stats["chunks"]
                else:
                    total_stats["skipped"] += 1

                total_stats["files"].append(stats)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                total_stats["failed"] += 1
                total_stats["files"].append({
                    "status": "failed",
                    "filename": pdf_path.name,
                    "error": str(e)
                })

        # Print summary
        logger.info("=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files: {total_stats['total_files']}")
        logger.info(f"Successful: {total_stats['successful']}")
        logger.info(f"Failed: {total_stats['failed']}")
        logger.info(f"Skipped: {total_stats['skipped']}")
        logger.info(f"Total pages: {total_stats['total_pages']}")
        logger.info(f"Total chunks: {total_stats['total_chunks']}")
        logger.info("=" * 60)

        return total_stats

    def get_collection_stats(self) -> Dict[str, any]:
        """Get current vector store statistics."""
        return self.vector_store.get_stats()
