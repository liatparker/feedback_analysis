# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Natural Language Queries)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LANGCHAIN AGENT                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  System Prompt + Tool Definitions                        │  │
│  │  - Decision making                                       │  │
│  │  - Tool selection                                        │  │
│  │  - Response generation                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TOOLS LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Semantic   │  │    Theme     │  │    Trend     │        │
│  │    Search    │  │   Analysis   │  │   Analysis   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Statistics  │  │    Reports   │  │  Historical  │        │
│  │   & Filters  │  │   Generation │  │   Retrieval  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   SHORT-TERM MEMORY      │  │   LONG-TERM MEMORY        │
│                          │  │                          │
│  ┌────────────────────┐ │  │  ┌────────────────────┐ │
│  │   Current CSV      │ │  │  │  Historical DB      │ │
│  │   (DataFrame)     │ │  │  │  (SQLite)          │ │
│  └────────────────────┘ │  │  └────────────────────┘ │
│                          │  │                          │
│  ┌────────────────────┐ │  │  ┌────────────────────┐ │
│  │   Vector DB       │ │  │  │  Complete Data     │ │
│  │   (ChromaDB)      │ │  │  │  (All Columns)     │ │
│  │   - Embeddings    │ │  │  │  - Indexed         │ │
│  │   - Metadata      │ │  │  │  - Time-based      │ │
│  └────────────────────┘ │  │  └────────────────────┘ │
└──────────────────────────┘  └──────────────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EMBEDDING SYSTEM                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SentenceTransformer Model                               │  │
│  │  - Text → Embedding Vector                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Three-Layer Caching                                     │  │
│  │  - Level 1: Hash Cache (exact match)                    │  │
│  │  - Level 2: Semantic Cache (similar match)              │  │
│  │  - Level 3: RAG Cache (query results)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Architecture

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INITIALIZATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  main.py: Load CSV                                                          │
│  └─ pandas.read_csv() → feedback_df                                        │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  initialize_tools(feedback_df)                                              │
│  ├─ Load Embedding Model (SentenceTransformer)                             │
│  ├─ _populate_vector_db()                                                  │
│  │   ├─ Check existing entries                                             │
│  │   ├─ For each entry:                                                    │
│  │   │   ├─ _generate_embeddings() [with caching]                         │
│  │   │   ├─ Check if in Vector DB                                          │
│  │   │   └─ Add if new (with metadata)                                     │
│  │   └─ Vector DB ready                                                    │
│  └─ _store_to_historical_db()                                              │
│      ├─ For each entry:                                                    │
│      │   ├─ Generate text hash                                             │
│      │   ├─ Check if hash exists                                           │
│      │   └─ Store if new (all columns as JSON)                             │
│      └─ Historical DB ready                                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUERY PROCESSING                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│   SHORT-TERM QUERY           │  │   LONG-TERM QUERY            │
│   (Current CSV)              │  │   (Historical Analysis)     │
└──────────────────────────────┘  └──────────────────────────────┘
            │                               │
            ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  semantic_search_feedback()  │  │  analyze_feedback_trends()   │
│  or                          │  │  or                          │
│  analyze_feedback_themes()   │  │  compare_feedback_periods()  │
└──────────────────────────────┘  └──────────────────────────────┘
            │                               │
            ▼                               ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│  CachedFeedbackRetriever     │  │  _get_historical_data()      │
│  └─ Query processing         │  │  └─ SQL query with filters   │
└──────────────────────────────┘  └──────────────────────────────┘
```

### Embedding Generation Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

Input: Text String
    │
    ▼
┌─────────────────────────────────┐
│  _normalize_text()              │
│  ├─ Lowercase                   │
│  ├─ Remove punctuation          │
│  └─ Normalize whitespace        │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  _hash_text(normalized)         │
│  └─ SHA256 hash                 │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 1: Hash Cache Check                                      │
│  └─ Key: text_{hash}                                            │
│      ├─ HIT: Return cached embedding (instant)                 │
│      └─ MISS: Continue                                         │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 2: Semantic Cache Check                                  │
│  └─ Check for exact normalized text match                      │
│      ├─ HIT: Return cached embedding                            │
│      └─ MISS: Generate embedding                               │
│          └─ model.encode([text])                               │
│              └─ Compare with cached embeddings (cosine > 0.95) │
│                  ├─ SIMILAR: Use cached                        │
│                  └─ NOT SIMILAR: Use generated                │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Save to Caches                                                 │
│  ├─ Level 1: Hash cache                                         │
│  └─ Level 2: Semantic cache                                     │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Optional: Add to Vector DB                                     │
│  └─ Only if:                                                    │
│      ├─ is_new_embedding = True                                │
│      ├─ add_to_vector_db = True                                │
│      └─ It's a feedback entry (not ad-hoc query)               │
└─────────────────────────────────────────────────────────────────┘
```

### Query Processing Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

User Query: "What are common complaints?"
    │
    ▼
┌─────────────────────────────────┐
│  LangChain Agent                │
│  └─ Decides: semantic_search    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  semantic_search_feedback(query="common complaints")            │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CachedFeedbackRetriever                                        │
│                                                                 │
│  LAYER 1: RAG Cache Check                                       │
│  └─ Key: rag_{query_hash}_{params_hash}                        │
│      ├─ HIT: Return cached results (instant)                   │
│      └─ MISS: Continue                                         │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Generate Query Embedding                              │
│  └─ _generate_embeddings([query])                              │
│      └─ Uses Level 1 & 2 caches (as shown above)              │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Semantic Query Cache Check                           │
│  └─ Compare query embedding with cached query embeddings       │
│      ├─ SIMILAR (>0.95): Return cached results                 │
│      └─ NOT SIMILAR: Continue                                  │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Vector DB Search                                      │
│  └─ Query Vector DB with query embedding                        │
│      ├─ Returns similar document embeddings                     │
│      ├─ Converts distances to similarities                      │
│      └─ Filters by similarity threshold                         │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: Cache Document Embeddings                             │
│  └─ For each retrieved document:                                │
│      └─ _generate_embeddings([doc_text])                       │
│          └─ Uses Level 1 & 2 caches                             │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 6: Store in RAG Cache                                    │
│  └─ Save: query + results + embeddings                          │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Return Results to Agent                                        │
│  └─ Documents with similarity scores                            │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent Analyzes Results                                         │
│  └─ Generates response for user                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA STORAGE LAYERS                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  SHORT-TERM MEMORY                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  In-Memory DataFrame (feedback_df)                                 │   │
│  │  - Current CSV data                                                │   │
│  │  - Fast access                                                     │   │
│  │  - Lost on restart                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Vector DB (ChromaDB) - .chroma_db/                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Collection: feedback_embeddings                            │   │   │
│  │  │  ├─ ID: feedback_{index}                                    │   │   │
│  │  │  ├─ Embedding: [0.23, -0.45, ...] (384-dim)                │   │   │
│  │  │  ├─ Document: Original text                                 │   │   │
│  │  │  └─ Metadata: {text, text_hash, level, timestamp}           │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  Purpose: Semantic search on current CSV                          │   │   │
│  │  Update: Only when new entries detected                           │   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  LONG-TERM MEMORY                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Historical DB (SQLite) - .historical_db/feedback_history.db       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Table: feedback_history                                    │   │   │
│  │  │  ├─ id: INTEGER PRIMARY KEY                                 │   │   │
│  │  │  ├─ text_hash: TEXT UNIQUE (duplicate detection)            │   │   │
│  │  │  ├─ stored_at: TIMESTAMP (when stored)                      │   │   │
│  │  │  ├─ feedback_data: TEXT (all columns as JSON)                │   │   │
│  │  │  ├─ level: INTEGER (indexed)                                │   │   │
│  │  │  └─ timestamp: TEXT (original timestamp, indexed)           │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  Indexes:                                                          │   │   │
│  │  ├─ idx_text_hash (duplicate detection)                           │   │   │
│  │  ├─ idx_stored_at (query by storage date)                         │   │   │
│  │  ├─ idx_timestamp (query by original timestamp)                   │   │   │
│  │  └─ idx_level (filter by level)                                  │   │   │
│  │  Purpose: Trend analysis over time                                │   │   │
│  │  Update: Only when new entries detected (hash-based)              │   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  CACHE LAYERS                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Level 1: Hash Cache - .embeddings_cache/exact_hash_cache.pkl     │   │
│  │  ├─ Key: text_{normalized_hash}                                   │   │
│  │  ├─ Value: Embedding vector                                        │   │
│  │  └─ Purpose: Instant lookup for identical texts                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Level 2: Semantic Cache - .embeddings_cache/semantic_cache.pkl   │   │
│  │  ├─ Key: semantic_{normalized_hash}                                │   │
│  │  ├─ Value: {embedding, timestamp}                                  │   │
│  │  ├─ Similarity: Cosine > 0.95                                     │   │
│  │  ├─ Eviction: LRU (max 10,000 entries)                            │   │
│  │  └─ Purpose: Reuse embeddings for very similar texts               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Level 3: RAG Cache - .embeddings_cache/rag_cache.pkl              │   │
│  │  ├─ Key: rag_{query_hash}_{params_hash}                             │   │
│  │  ├─ Value: {result, query_embedding, document_embeddings, timestamp}│   │
│  │  ├─ Eviction: LRU (max 5,000 entries)                              │   │
│  │  └─ Purpose: Cache complete query results                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY SYSTEM FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

CSV File Loaded
    │
    ▼
┌─────────────────────────────────┐
│  Load into DataFrame            │
│  └─ feedback_df (in-memory)     │
└─────────────┬───────────────────┘
              │
              ├──────────────────────────────┐
              │                              │
              ▼                              ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  SHORT-TERM PATH        │  │  LONG-TERM PATH         │
│                         │  │                         │
│  Populate Vector DB     │  │  Store to Historical DB │
│  └─ For each entry:    │  │  └─ For each entry:    │
│      ├─ Generate       │  │      ├─ Generate       │
│      │  embedding      │  │      │  text hash      │
│      ├─ Check if       │  │      ├─ Check if       │
│      │  exists         │  │      │  hash exists    │
│      └─ Add if new     │  │      └─ Store if new   │
│                         │  │                         │
│  Ready for:            │  │  Ready for:            │
│  - Semantic search     │  │  - Trend analysis      │
│  - Current queries     │  │  - Period comparison   │
│  - Immediate analysis  │  │  - Long-term patterns  │
└─────────────────────────┘  └─────────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             │
                             ▼
                    User Query Arrives
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Current Document    │  │  Trend Analysis      │
    │  Questions           │  │  Questions           │
    │                      │  │                      │
    │  Use:                │  │  Use:                │
    │  - feedback_df       │  │  - Historical DB     │
    │  - Vector DB         │  │  - get_historical_   │
    │  - Default tools     │  │    feedback()        │
    │                      │  │  - use_historical=  │
    │  Examples:           │  │    True              │
    │  - "What are         │  │                      │
    │    complaints?"      │  │  Examples:           │
    │  - "Show level 3"    │  │  - "Trends over      │
    │                      │  │    6 months?"        │
    └──────────────────────┘  │  - "Compare periods" │
                               └──────────────────────┘
```

## Key Design Principles

1. **Separation of Concerns**
   - Short-term memory: Current CSV analysis
   - Long-term memory: Historical trend analysis

2. **Layered Caching**
   - Level 1: Exact hash match (fastest)
   - Level 2: Semantic similarity (reuse similar)
   - Level 3: Query results (complete pipeline)

3. **Efficient Updates**
   - Duplicate detection prevents redundant storage
   - Incremental updates only for new entries
   - Cache-first approach minimizes computation

4. **Scalability**
   - Vector DB for fast semantic search
   - SQLite for efficient structured queries
   - Caching reduces embedding generation overhead

