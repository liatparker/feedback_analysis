# Data Flow Documentation

## Overview

This document describes the data flow through the Feedback Analyzer system, from CSV loading to query processing and storage.

## Initialization Flow

### 1. CSV File Loading
```
User loads CSV file
  ↓
main.py reads CSV into pandas DataFrame (feedback_df)
  ↓
initialize_tools(feedback_df) is called
```

### 2. Tool Initialization
```
initialize_tools(dataframe)
  ↓
├─ Load embedding model (SentenceTransformer)
  ↓
├─ SHORT-TERM MEMORY: Populate Vector DB
│   └─ _populate_vector_db()
│       ├─ Check existing entries in Vector DB
│       ├─ For each feedback entry:
│       │   ├─ Generate embedding (with caching)
│       │   ├─ Check if already in Vector DB
│       │   └─ Add if new (with metadata: text, level, timestamp)
│       └─ Vector DB ready for semantic search
  ↓
└─ LONG-TERM MEMORY: Store to Historical DB
    └─ _store_to_historical_db(dataframe)
        ├─ For each feedback entry:
        │   ├─ Generate text hash (normalized)
        │   ├─ Check if hash exists in Historical DB
        │   └─ Store if new (all columns as JSON)
        └─ Historical DB ready for trend analysis
```

## Query Processing Flow

### Short-Term Query (Current CSV)

```
User Query: "What are common complaints?"
  ↓
Agent decides to use semantic_search_feedback
  ↓
semantic_search_feedback(query="common complaints")
  ↓
├─ Use CachedFeedbackRetriever
│   ├─ LAYER 1: Check RAG Cache
│   │   └─ If found: Return cached results
│   │
│   ├─ LAYER 2: Generate Query Embedding
│   │   └─ _generate_embeddings([query])
│   │       ├─ Level 1: Check hash cache (exact match)
│   │       ├─ Level 2: Check semantic cache (similar match)
│   │       └─ If miss: Generate embedding
│   │
│   ├─ LAYER 3: Check Semantic Query Cache
│   │   └─ If similar query found: Return cached results
│   │
│   ├─ LAYER 4: Vector DB Search
│   │   └─ Query Vector DB with query embedding
│   │       └─ Returns similar document embeddings
│   │
│   ├─ LAYER 5: Cache Document Embeddings
│   │   └─ Cache retrieved document embeddings
│   │
│   └─ LAYER 6: Store in RAG Cache
│       └─ Save query + results for next time
  ↓
Results returned to agent
  ↓
Agent analyzes results and responds to user
```

### Long-Term Query (Historical Analysis)

```
User Query: "How have complaints changed over 6 months?"
  ↓
Agent decides to use analyze_feedback_trends(use_historical=True)
  ↓
analyze_feedback_trends(use_historical=True, start_date="...", end_date="...")
  ↓
├─ Retrieve from Historical DB
│   └─ _get_historical_data(start_date, end_date)
│       ├─ SQL query with date filters
│       ├─ Parse JSON data back to DataFrame
│       └─ Return historical feedback
  ↓
├─ Process historical data
│   ├─ Group by time period (day/week/month/quarter/year)
│   ├─ Calculate statistics per period
│   └─ Compare periods if requested
  ↓
Results returned to agent
  ↓
Agent analyzes trends and responds to user
```

## Embedding Generation Flow

### Single Text Embedding

```
Input: Text string
  ↓
_normalize_text(text)
  ├─ Lowercase
  ├─ Remove punctuation
  └─ Normalize whitespace
  ↓
_hash_text(normalized_text)
  └─ SHA256 hash of normalized text
  ↓
Check Level 1 Cache (Hash Cache)
  ├─ If found: Return cached embedding
  └─ If not found: Continue
  ↓
Check Level 2 Cache (Semantic Cache)
  ├─ Check for exact normalized text match
  │   └─ If found: Return cached embedding
  └─ If not found: Generate embedding
      └─ model.encode([text])
  ↓
Save to both caches
  ├─ Level 1: Hash cache
  └─ Level 2: Semantic cache
  ↓
Return embedding
```

### Batch Text Embeddings

```
Input: List of text strings
  ↓
For each text:
  ├─ Check Level 1 cache
  └─ If miss: Add to generation list
  ↓
Generate embeddings for misses (batch)
  └─ model.encode(texts_to_generate)
  ↓
Cache all new embeddings
  ↓
Reconstruct full embedding array (cached + new)
  ↓
Return embeddings
```

## Vector DB Update Flow

### New Entry Detection

```
New CSV loaded with additional entries
  ↓
_populate_vector_db()
  ↓
Check existing count in Vector DB
  ├─ If count matches: Skip (no new entries)
  └─ If count differs: Process new entries
      ↓
      For each entry:
      ├─ Check if ID exists in Vector DB
      ├─ If exists: Skip (duplicate)
      └─ If new:
          ├─ Generate embedding (with caching)
          ├─ Prepare metadata (text, level, timestamp)
          └─ Add to Vector DB
```

## Historical DB Update Flow

### New Entry Storage

```
New CSV loaded
  ↓
_store_to_historical_db(dataframe)
  ↓
For each feedback entry:
  ├─ Generate normalized text hash
  ├─ Check if hash exists in Historical DB
  ├─ If exists: Skip (duplicate)
  └─ If new:
      ├─ Convert all columns to JSON
      ├─ Extract level and timestamp for indexing
      └─ Insert into Historical DB
```

## Caching Layers

### Level 1: Hash Cache (Exact Match)
- Purpose: Instant lookup for identical texts
- Key: Normalized text hash
- Storage: Pickle file
- Lookup: O(1) dictionary

### Level 2: Semantic Cache (Similar Match)
- Purpose: Reuse embeddings for very similar texts (>95% similarity)
- Key: Normalized text hash
- Storage: Pickle file with LRU eviction
- Lookup: Cosine similarity comparison

### Level 3: RAG Cache (Query Results)
- Purpose: Cache complete query results
- Key: Query hash + parameters hash
- Storage: Pickle file with LRU eviction
- Lookup: Query embedding similarity

## Storage Systems

### Vector DB (ChromaDB)
- Purpose: Semantic search on current CSV
- Storage: .chroma_db/ directory
- Contents: Embeddings + metadata (text, level, timestamp)
- Update: Only when new entries detected

### Historical DB (SQLite)
- Purpose: Long-term trend analysis
- Storage: .historical_db/feedback_history.db
- Contents: All columns as JSON + indexes (timestamp, level, text_hash)
- Update: Only when new entries detected (duplicate detection by text hash)

### Cache Files
- Level 1: .embeddings_cache/exact_hash_cache.pkl
- Level 2: .embeddings_cache/semantic_cache.pkl
- Level 3: .embeddings_cache/rag_cache.pkl

## Key Design Principles

1. Cache First: Always check caches before generating embeddings
2. Duplicate Detection: Both Vector DB and Historical DB check for duplicates
3. Incremental Updates: Only new entries are added, not full repopulation
4. Layered Caching: Three levels of caching for optimal performance
5. Separation of Concerns: Short-term (Vector DB) vs Long-term (Historical DB)

