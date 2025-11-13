# Feedback Analyzer

A LangChain-based agent system for analyzing feedback data with semantic search, theme analysis, and long-term trend tracking.

## Quick Start

### Prerequisites
- Python 3.8+
- Anthropic API key (or OpenAI API key)

### Installation

1. Clone or navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the `FeedBack_Analyzer` directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Or for OpenAI:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Update the input file path in `main.py`:
```python
INPUT_FILE = '/path/to/your/feedback_data.csv'
```

5. Run the application:
```bash
python main.py
```

## Configuration

Edit the configuration section in `main.py`:

- `INPUT_FILE`: Path to your CSV file
- `LLM_MODEL`: Model name (e.g., "claude-sonnet-4-5-20250929")
- `LLM_PROVIDER`: "anthropic" or "openai"
- `LLM_TEMPERATURE`: Model temperature (default: 0)
- `VERBOSE`: Set to True for detailed agent execution logs

## CSV File Format

Your CSV file should contain:
- **Level column**: Numeric rating/ranking (e.g., 1-5, 1-10)
- **Text column**: Free-style text feedback (column name: "text")
- **Optional columns**: Timestamp, ID, reference numbers, etc.

Example:
```
Level,Text,Timestamp
5,"Great service, very helpful",2024-01-15
3,"App is slow sometimes",2024-01-16
```

## System Architecture

### Two-Tier Memory System

**Short-Term Memory (Current CSV)**
- Purpose: Immediate analysis of current document
- Storage: In-memory DataFrame + Vector DB for semantic search
- Use: Questions about current CSV file

**Long-Term Memory (Historical Database)**
- Purpose: Trend analysis across multiple CSV files
- Storage: SQLite database accumulating data over time
- Use: Questions about trends, changes, period comparisons

### Key Components

**Embedding System**
- Uses sentence-transformers for text embeddings
- Three-layer caching: Hash cache, semantic cache, RAG cache
- Vector DB (ChromaDB) for semantic search

**Analysis Tools**
- Semantic search: Find similar feedback entries
- Theme analysis: Clustering or LLM-based categorization
- Trend analysis: Time-based patterns and comparisons
- Statistical reports: CSV exports with filtering

## Usage Examples

### Current Document Analysis
```
User: "What are common complaints in this file?"
Agent: Uses semantic search on current CSV, analyzes results
```

### Trend Analysis
```
User: "How have complaints changed over the last 6 months?"
Agent: Uses historical database, analyzes trends by period
```

### Period Comparison
```
User: "Compare feedback from January vs February"
Agent: Retrieves historical data for both periods, compares statistics
```

## Tools Available

1. **get_feedback_by_level**: Filter feedback by specific level
2. **get_feedback_statistics**: Get overall statistics about feedback
3. **generate_dataframe_report**: Generate filtered CSV reports
4. **semantic_search_feedback**: Semantic search for similar entries
5. **analyze_feedback_themes**: Theme analysis with clustering or LLM
6. **generate_categorized_issues_report**: Categorized CSV reports
7. **get_historical_feedback**: Retrieve historical feedback data
8. **analyze_feedback_trends**: Analyze trends over time
9. **compare_feedback_periods**: Compare two time periods

## Data Storage

**Vector DB** (`.chroma_db/`)
- Stores embeddings for semantic search
- Updated only when new entries detected
- Uses smart caching to avoid recomputing embeddings

**Historical DB** (`.historical_db/`)
- Stores complete feedback data for trend analysis
- Automatically accumulates data from loaded CSV files
- Duplicate detection prevents storing same entry twice

**Cache Files** (`.embeddings_cache/`)
- Level 1: Exact hash cache
- Level 2: Semantic similarity cache
- Level 3: RAG cache (query results)

## Performance Optimizations

- **Layered Caching**: Three levels of caching to avoid redundant embedding generation
- **Duplicate Detection**: Prevents storing same feedback multiple times
- **Incremental Updates**: Only new entries are processed
- **Batch Processing**: Efficient handling of large datasets

## File Structure

```
FeedBack_Analyzer/
├── main.py              # Main application entry point
├── tools.py             # Tool definitions and core logic
├── retriever.py         # Custom LangChain retriever
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── DATA_FLOW.md         # Detailed data flow documentation
└── .env                 # Environment variables (create this)
```

## Troubleshooting

**API Key Error**
- Ensure `.env` file exists with correct API key
- Check API key is valid and has proper permissions

**Model Not Found**
- Verify model name matches available models in your API account
- Check model name spelling and version

**No Data Found**
- Verify CSV file path is correct
- Check CSV has required columns (Level, Text)
- Ensure CSV file is readable

**Performance Issues**
- Cache files are created automatically on first run
- Subsequent runs will be faster due to caching
- Large datasets may take time on first load

## Advanced Configuration

**Embedding Model**
- Default: `paraphrase-multilingual-MiniLM-L12-v2`
- Can be changed in `tools.py` `_get_embedding_model()`

**Cache Settings**
- Cache sizes and TTL can be adjusted in `tools.py`
- Similarity thresholds for semantic cache can be modified

**Vector DB Settings**
- ChromaDB settings in `tools.py` `_initialize_vector_db()`
- Collection name and metadata can be customized

## Notes

- Historical data is automatically stored when CSV files are loaded
- Only new entries are added to avoid duplicates
- Vector DB is used for semantic search on current CSV
- Historical DB is used for trend analysis over time
- All caching is transparent and automatic

