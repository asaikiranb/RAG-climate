# CLI Scripts

Command-line tools for document ingestion and search.

## Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. Document Ingestion (`ingest.py`)

Ingest PDF documents into the vector database.

#### Basic Usage

```bash
# Ingest all PDFs from a folder
python scripts/ingest.py data/pdfs/

# Ingest a single PDF
python scripts/ingest.py data/pdfs/document.pdf

# Show current collection statistics
python scripts/ingest.py --stats
```

#### Advanced Options

```bash
# Reset collection and re-ingest
python scripts/ingest.py data/pdfs/ --reset

# Verbose output
python scripts/ingest.py data/pdfs/ -v
```

#### Output

The script provides:
- Progress reporting for each file
- Summary statistics (files processed, pages, chunks created)
- Collection statistics after ingestion
- Error handling with detailed messages

Example output:
```
======================================================================
  DOCUMENT INGESTION
======================================================================
Collection: hvac_documents
Embedding:  all-MiniLM-L6-v2
Chunking:   token (1000 tokens)

Ingesting folder: data/pdfs/
Found 5 PDF files...

======================================================================
  INGESTION SUMMARY
======================================================================
  Total Files:       5
  Successful:        5
  Failed:            0
  Skipped:           0
  Total Pages:       150
  Total Chunks:      450
  Avg Chunks/File:   90.0
======================================================================
```

---

### 2. Search Tool (`search.py`)

Search the document collection using hybrid retrieval.

#### Basic Usage

```bash
# Basic search
python scripts/search.py "What is the Montreal Protocol?"

# Return more results
python scripts/search.py "low-GWP refrigerants" --top-k 10

# Show full document text
python scripts/search.py "passive cooling strategies" --full
```

#### Advanced Options

```bash
# Disable advanced features for faster search
python scripts/search.py "HVAC maintenance" --simple

# Table format display
python scripts/search.py "refrigerant alternatives" --table

# Export results to JSON
python scripts/search.py "India Cooling Action Plan" --export results.json

# Include detailed retrieval metadata
python scripts/search.py "cooling technologies" --metadata

# Verbose output
python scripts/search.py "energy efficiency" -v
```

#### Output Formats

**Default format** (detailed):
```
[1] Score: 0.8945
    File:  India_Cooling_Action_Plan.pdf
    Page:  5
    Retrieved with: 3 query variations
    Reranked: Yes

The India Cooling Action Plan (ICAP) aims to reduce cooling demand...
--------------------------------------------------------------------------------
```

**Table format** (`--table`):
```
--------------------------------------------------------------------------------
#    Score    File                           Page   Preview
--------------------------------------------------------------------------------
1    0.8945   India_Cooling_Action_Plan.pdf  5      The India Cooling Action...
2    0.8532   Montreal_Protocol_Guide.pdf    12     The Montreal Protocol is...
--------------------------------------------------------------------------------
```

**JSON export** (`--export results.json`):
```json
[
  {
    "document": "Full document text...",
    "metadata": {
      "filename": "India_Cooling_Action_Plan.pdf",
      "page_number": "5",
      "score": 0.8945
    },
    "retrieval_metadata": {
      "query_variations": 3,
      "reranked": true,
      "diversified": true
    }
  }
]
```

---

## Configuration

Both scripts use the centralized configuration from `src/config/settings.py`. Configuration can be customized via:

1. **Environment variables** (`.env` file):
   ```bash
   GROQ_API_KEY=your_api_key
   CHROMA_COLLECTION_NAME=hvac_documents
   ```

2. **Default settings** in `src/config/settings.py`

### Key Settings

- **Embedding model**: `all-MiniLM-L6-v2` (default)
- **Chunking**: Token-based with 1000 token chunks, 200 overlap
- **Retrieval**: Hybrid search with vector + BM25
- **Advanced features**: HyDE, query expansion, reranking, MMR

## Error Handling

Both scripts include comprehensive error handling:

- File not found errors
- Collection initialization errors
- Ingestion failures (logged with details)
- Search failures (with stack traces in verbose mode)

Errors are logged to `logs/rag.log` for debugging.

## Examples

### Complete Workflow

```bash
# 1. Check current collection stats
python scripts/ingest.py --stats

# 2. Ingest new documents
python scripts/ingest.py data/new_pdfs/

# 3. Search the collection
python scripts/search.py "What are the latest cooling technologies?" --top-k 5

# 4. Export results for analysis
python scripts/search.py "refrigerant regulations" --export analysis.json

# 5. Quick search without advanced features
python scripts/search.py "HVAC troubleshooting" --simple --table
```

### Reset and Re-ingest

```bash
# Clear existing collection and ingest fresh data
python scripts/ingest.py data/pdfs/ --reset
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### ChromaDB Connection Issues

Check that the collection exists:
```bash
python scripts/ingest.py --stats
```

If the collection doesn't exist, create it by ingesting documents:
```bash
python scripts/ingest.py data/pdfs/
```

### API Key Errors

Ensure your `.env` file contains:
```bash
GROQ_API_KEY=your_api_key_here
```

## Performance Tips

1. **Use `--simple` for faster searches** when you don't need advanced features
2. **Adjust `--top-k`** to retrieve fewer/more results
3. **Use `--table` format** for quick scanning of results
4. **Enable `--metadata`** only when you need detailed retrieval information
