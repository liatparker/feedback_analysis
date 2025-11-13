"""
Tools for feedback analysis agent.
These tools provide functionality to query and analyze feedback data.
Includes embedding-based semantic search, clustering, and similarity analysis.
"""

from langchain_core.tools import tool
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import os
import pickle
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import hashlib
import re
from collections import OrderedDict
from datetime import datetime, timedelta
import sqlite3

# Global variables
feedback_df = None
_embedding_model = None
_embeddings_cache = None  # Level 1: Exact hash cache
_semantic_cache = None     # Level 2: Semantic similarity cache
_rag_cache = None          # Level 3: RAG cache (query → results)
_embeddings_file = None
_semantic_cache_file = None
_rag_cache_file = None
_vector_db = None
_vector_collection = None
_historical_db_path = None  # Long-term memory: Historical feedback data

# Cache configuration
CACHE_SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold for semantic cache
CACHE_MAX_SIZE = 10000  # Maximum entries in semantic cache (LRU eviction)
CACHE_TTL_HOURS = 24  # Time-to-live for cache entries
RAG_CACHE_MAX_SIZE = 5000  # Maximum entries in RAG cache (LRU eviction)
RAG_CACHE_TTL_HOURS = 24  # Time-to-live for RAG cache entries

# Normalization configuration (optimized for free-style feedback text)
NORMALIZE_REMOVE_PUNCTUATION = True  # Remove punctuation for better cache hits (e.g., "slow!" = "slow")
NORMALIZE_USE_STEMMING = False  # Disabled: too aggressive for feedback, slower, may lose meaning
NORMALIZE_NORMALIZE_WHITESPACE = True  # Always enabled: normalize multiple spaces/tabs/newlines

def _get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        # Using a multilingual model that works well for various languages
        _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model

def _load_embeddings_cache():
    """
    Load Level 1 cache: Exact hash cache (fast, memory-based).
    This is the primary cache for exact text matches.
    """
    global _embeddings_cache, _embeddings_file
    if _embeddings_cache is None:
        cache_dir = os.path.join(os.getcwd(), '.embeddings_cache')
        os.makedirs(cache_dir, exist_ok=True)
        _embeddings_file = os.path.join(cache_dir, 'exact_hash_cache.pkl')
        
        if os.path.exists(_embeddings_file):
            try:
                with open(_embeddings_file, 'rb') as f:
                    _embeddings_cache = pickle.load(f)
            except:
                _embeddings_cache = {}
        else:
            _embeddings_cache = {}
    return _embeddings_cache

def _load_semantic_cache():
    """
    Load Level 2 cache: Semantic similarity cache (vector-based).
    Uses LRU eviction and TTL for cache freshness.
    """
    global _semantic_cache, _semantic_cache_file
    if _semantic_cache is None:
        cache_dir = os.path.join(os.getcwd(), '.embeddings_cache')
        os.makedirs(cache_dir, exist_ok=True)
        _semantic_cache_file = os.path.join(cache_dir, 'semantic_cache.pkl')
        
        if os.path.exists(_semantic_cache_file):
            try:
                with open(_semantic_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Restore OrderedDict for LRU behavior
                    _semantic_cache = OrderedDict(cache_data)
                    # Clean expired entries
                    _clean_expired_cache_entries()
            except:
                _semantic_cache = OrderedDict()
        else:
            _semantic_cache = OrderedDict()
    return _semantic_cache

def _clean_expired_cache_entries():
    """Remove expired entries from semantic cache based on TTL."""
    global _semantic_cache
    if not _semantic_cache:
        return
    
    current_time = datetime.now()
    expired_keys = []
    
    for key, value in _semantic_cache.items():
        if isinstance(value, dict) and 'timestamp' in value:
            entry_time = value['timestamp']
            if isinstance(entry_time, datetime):
                age = current_time - entry_time
                if age > timedelta(hours=CACHE_TTL_HOURS):
                    expired_keys.append(key)
    
    for key in expired_keys:
        _semantic_cache.pop(key, None)
    
    if expired_keys:
        _save_semantic_cache()

def _save_embeddings_cache():
    """Save Level 1 cache (exact hash cache) to disk."""
    global _embeddings_cache, _embeddings_file
    if _embeddings_cache and _embeddings_file:
        try:
            with open(_embeddings_file, 'wb') as f:
                pickle.dump(_embeddings_cache, f)
        except Exception as e:
            print(f"Warning: Could not save embeddings cache: {e}")

def _save_semantic_cache():
    """Save Level 2 cache (semantic similarity cache) to disk."""
    global _semantic_cache, _semantic_cache_file
    if _semantic_cache and _semantic_cache_file:
        try:
            # Convert OrderedDict to regular dict for serialization
            cache_data = dict(_semantic_cache)
            with open(_semantic_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not save semantic cache: {e}")

def _find_similar_in_semantic_cache(query_embedding: np.ndarray, threshold: float = CACHE_SIMILARITY_THRESHOLD) -> Optional[np.ndarray]:
    """
    Level 2 cache check: Find semantically similar embedding in cache.
    
    Args:
        query_embedding: Embedding vector to search for
        threshold: Cosine similarity threshold (default: 0.95)
    
    Returns:
        Cached embedding if found, None otherwise
    """
    semantic_cache = _load_semantic_cache()
    if not semantic_cache:
        return None
    
    # Convert query embedding to 2D array for cosine similarity
    query_emb_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
    
    # Check each cached embedding
    for key, cache_entry in semantic_cache.items():
        if isinstance(cache_entry, dict) and 'embedding' in cache_entry:
            cached_emb = cache_entry['embedding']
            cached_emb_2d = cached_emb.reshape(1, -1) if cached_emb.ndim == 1 else cached_emb
            
            # Compute cosine similarity
            similarity = cosine_similarity(query_emb_2d, cached_emb_2d)[0][0]
            
            if similarity >= threshold:
                # Update access time for LRU
                semantic_cache.move_to_end(key)
                return cached_emb
    
    return None

def _add_to_semantic_cache(text: str, embedding: np.ndarray):
    """
    Add embedding to Level 2 semantic cache with LRU eviction.
    
    Args:
        text: Original text
        embedding: Embedding vector
    """
    global _semantic_cache
    semantic_cache = _load_semantic_cache()
    
    # Use text hash as key
    text_hash = _hash_text(text, normalized=True)
    key = f"semantic_{text_hash}"
    
    # Add entry with timestamp
    semantic_cache[key] = {
        'text': text,
        'embedding': embedding,
        'timestamp': datetime.now()
    }
    
    # LRU eviction: remove oldest if cache is too large
    while len(semantic_cache) > CACHE_MAX_SIZE:
        semantic_cache.popitem(last=False)  # Remove oldest (first) item
    
    _save_semantic_cache()

def _load_rag_cache():
    """Load Level 3 cache: RAG cache (query → results) with LRU and TTL."""
    global _rag_cache, _rag_cache_file
    if _rag_cache is None:
        _rag_cache = OrderedDict()
        if _rag_cache_file is None:
            cache_dir = os.path.join(os.getcwd(), '.cache')
            os.makedirs(cache_dir, exist_ok=True)
            _rag_cache_file = os.path.join(cache_dir, 'rag_cache.pkl')
        
        if os.path.exists(_rag_cache_file):
            try:
                with open(_rag_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    _rag_cache = OrderedDict(cache_data)
                # Clean expired entries on load
                _clean_expired_rag_cache_entries()
            except Exception as e:
                print(f"Warning: Could not load RAG cache: {e}")
                _rag_cache = OrderedDict()
    
    return _rag_cache

def _save_rag_cache():
    """Save Level 3 cache (RAG cache) to disk."""
    global _rag_cache, _rag_cache_file
    if _rag_cache and _rag_cache_file:
        try:
            # Convert OrderedDict to regular dict for serialization
            cache_data = dict(_rag_cache)
            with open(_rag_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not save RAG cache: {e}")

def _clean_expired_rag_cache_entries():
    """Remove expired entries from RAG cache based on TTL."""
    global _rag_cache
    if not _rag_cache:
        return
    
    current_time = datetime.now()
    expired_keys = []
    
    for key, value in _rag_cache.items():
        if isinstance(value, dict) and 'timestamp' in value:
            entry_time = value['timestamp']
            if isinstance(entry_time, datetime):
                age = current_time - entry_time
                if age > timedelta(hours=RAG_CACHE_TTL_HOURS):
                    expired_keys.append(key)
    
    for key in expired_keys:
        _rag_cache.pop(key, None)
    
    if expired_keys:
        _save_rag_cache()

def _get_rag_cache_key(query: str, top_k: int, level_filter: int = None, min_level: int = None, max_level: int = None, filter_by_level: bool = False, similarity_threshold: float = 0.3) -> str:
    """
    Generate RAG cache key from query and parameters.
    Uses normalized query hash + parameter hash for consistent caching.
    
    Args:
        query: Query string
        top_k: Number of results
        level_filter: Level filter value
        min_level: Min level filter
        max_level: Max level filter
        filter_by_level: Whether to filter by level
        similarity_threshold: Similarity threshold
    
    Returns:
        Cache key string
    """
    # Normalize query and hash it
    query_hash = _hash_text(query, normalized=True)
    
    # Create parameter hash (to handle different search parameters)
    params = f"k{top_k}_f{filter_by_level}_t{similarity_threshold}"
    if level_filter is not None:
        params += f"_lf{level_filter}"
    if min_level is not None:
        params += f"_min{min_level}"
    if max_level is not None:
        params += f"_max{max_level}"
    
    param_hash = hashlib.sha256(params.encode('utf-8')).hexdigest()[:8]
    
    return f"rag_{query_hash}_{param_hash}"

def _get_rag_cache_result(cache_key: str) -> Optional[str]:
    """
    Get cached RAG result if available and not expired.
    
    Args:
        cache_key: RAG cache key
    
    Returns:
        Cached result string if found and valid, None otherwise
    """
    rag_cache = _load_rag_cache()
    if cache_key in rag_cache:
        entry = rag_cache[cache_key]
        if isinstance(entry, dict) and 'result' in entry:
            # Check TTL
            if 'timestamp' in entry:
                entry_time = entry['timestamp']
                if isinstance(entry_time, datetime):
                    age = datetime.now() - entry_time
                    if age <= timedelta(hours=RAG_CACHE_TTL_HOURS):
                        # Update LRU (move to end)
                        rag_cache.move_to_end(cache_key)
                        return entry['result']
                    else:
                        # Expired, remove it
                        rag_cache.pop(cache_key, None)
                        _save_rag_cache()
            else:
                # No timestamp, assume valid
                rag_cache.move_to_end(cache_key)
                return entry['result']
    return None

def _store_rag_cache_result(cache_key: str, result: str, query_embedding: np.ndarray = None, document_embeddings: List[np.ndarray] = None):
    """
    Store RAG result in cache with LRU eviction.
    Stores both query and document embeddings for semantic cache lookup.
    
    Args:
        cache_key: RAG cache key
        result: Result string to cache
        query_embedding: Query embedding vector (optional, for semantic cache)
        document_embeddings: List of document embedding vectors (optional, for caching document embeddings)
    """
    global _rag_cache
    rag_cache = _load_rag_cache()
    
    # Add entry with timestamp and optional embeddings
    entry = {
        'result': result,
        'timestamp': datetime.now()
    }
    if query_embedding is not None:
        entry['query_embedding'] = query_embedding
    if document_embeddings is not None:
        entry['document_embeddings'] = document_embeddings
    
    rag_cache[cache_key] = entry
    
    # LRU eviction: remove oldest if cache is too large
    while len(rag_cache) > RAG_CACHE_MAX_SIZE:
        rag_cache.popitem(last=False)  # Remove oldest (first) item
    
    _save_rag_cache()

def _find_similar_query_in_rag_cache(query_embedding: np.ndarray, threshold: float = CACHE_SIMILARITY_THRESHOLD) -> Optional[str]:
    """
    Check RAG cache for similar query embeddings (query vs previous queries).
    This enables finding cached answers for semantically similar queries.
    
    Args:
        query_embedding: Query embedding vector
        threshold: Cosine similarity threshold (default: 0.95)
    
    Returns:
        Cached result string if similar query found, None otherwise
    """
    rag_cache = _load_rag_cache()
    if not rag_cache:
        return None
    
    # Convert query embedding to 2D array for cosine similarity
    query_emb_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
    
    # Check each cached entry for similar query embedding
    for key, cache_entry in rag_cache.items():
        if isinstance(cache_entry, dict) and 'query_embedding' in cache_entry:
            cached_emb = cache_entry['query_embedding']
            cached_emb_2d = cached_emb.reshape(1, -1) if cached_emb.ndim == 1 else cached_emb
            
            # Compute cosine similarity
            similarity = cosine_similarity(query_emb_2d, cached_emb_2d)[0][0]
            
            if similarity >= threshold:
                # Found similar query - return cached result
                if 'result' in cache_entry:
                    # Update LRU
                    rag_cache.move_to_end(key)
                    return cache_entry['result']
    
    return None

def _format_text_for_display(text: str, max_length: int = None) -> str:
    """
    Format text for display, preserving original language direction.
    Does not force RTL/LTR - displays text in its natural language direction.
    
    Args:
        text: Text to format
        max_length: Maximum length (truncate if longer)
    
    Returns:
        Formatted text (truncated if needed, but preserving natural direction)
    """
    if not text or pd.isna(text):
        return ""
    
    text_str = str(text)
    
    # Truncate if needed
    if max_length and len(text_str) > max_length:
        text_str = text_str[:max_length] + "..."
    
    # Return text as-is, preserving its natural language direction
    # The terminal/console will handle RTL/LTR automatically based on the text content
    return text_str

def _get_text_column(df: pd.DataFrame) -> str:
    """
    Get the text column name from the dataframe.
    The column is fixed as 'text' (stable, permanent header).
    
    Args:
        df: DataFrame with 'text' column
    
    Returns:
        'text' if column exists, None otherwise
    """
    # Fixed column name - no need for flexible searching
    if 'text' in df.columns:
        return 'text'
    
    # Case-insensitive check as fallback
    for col in df.columns:
        if col.lower() == 'text':
            return col
    
    return None

def _get_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find timestamp/date column in the dataframe.
    Checks common column names for timestamps.
    
    Args:
        df: DataFrame to search
    
    Returns:
        Column name if found, None otherwise
    """
    # Common timestamp column names (case-insensitive)
    timestamp_names = ['timestamp', 'date', 'created_at', 'created', 'time', 
                      'datetime', 'date_time', 'submitted_at', 'received_at',
                      'feedback_date', 'entry_date']
    
    # First check exact match (case-sensitive)
    for name in timestamp_names:
        if name in df.columns:
            return name
    
    # Then check case-insensitive
    for col in df.columns:
        if col.lower() in [name.lower() for name in timestamp_names]:
            return col
    
    return None

def _normalize_text(text: str, use_stemming: bool = False, remove_punctuation: bool = True, normalize_whitespace: bool = True) -> str:
    """
    Efficient text normalization optimized for free-style feedback text.
    
    Designed for user feedback which may contain:
    - Typos and informal language
    - Varying punctuation (e.g., "slow!" vs "slow")
    - Inconsistent whitespace
    - Mixed case
    
    Args:
        text: Text to normalize
        use_stemming: If True, apply stemming (default: False, too aggressive for feedback)
        remove_punctuation: If True, remove punctuation (default: True, helps catch variations)
        normalize_whitespace: If True, normalize whitespace (default: True, always recommended)
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Step 1: Lowercase (fast, always beneficial)
    normalized = text.lower()
    
    # Step 2: Normalize whitespace (fast, handles tabs/newlines/multiple spaces)
    if normalize_whitespace:
        # Replace all whitespace (spaces, tabs, newlines) with single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Step 3: Remove punctuation (fast, helps catch "slow!" vs "slow" vs "slow?")
    # This is beneficial for free-style text where punctuation varies
    if remove_punctuation:
        # Remove punctuation but keep alphanumeric and spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Clean up any extra spaces created by punctuation removal
        normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Step 4: Stemming (DISABLED by default - too aggressive for feedback)
    # Stemming would change "running" -> "run", "better" -> "better" (no change)
    # This can lose important nuances in feedback (e.g., "improving" vs "improved")
    # Also slower and requires nltk dependency
    if use_stemming:
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            words = normalized.split()
            normalized = ' '.join([stemmer.stem(word) for word in words])
        except ImportError:
            # nltk not available, skip stemming
            pass
        except Exception:
            # Stemming failed, continue without it
            pass
    
    return normalized

def _hash_text(text: str, normalized: bool = True) -> str:
    """
    Generate a consistent hash for text content.
    
    Args:
        text: Text to hash
        normalized: If True, normalize text first using configured normalization (default: True)
    
    Returns:
        16-character hex hash
    """
    if normalized:
        text = _normalize_text(
            text, 
            use_stemming=NORMALIZE_USE_STEMMING,
            remove_punctuation=NORMALIZE_REMOVE_PUNCTUATION,
            normalize_whitespace=NORMALIZE_NORMALIZE_WHITESPACE
        )
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def _generate_embeddings(texts: List[str], use_smart_cache: bool = True, use_semantic_cache: bool = True, add_to_vector_db: bool = False, vector_db_indices: List[int] = None, level_values: List[int] = None) -> np.ndarray:
    """
    Generate embeddings for a list of texts with layered caching strategy.
    
    OPTIMAL CACHING ARCHITECTURE:
    1. Level 1: Exact hash cache (fast, memory-based) - check normalized text hash
    2. Level 2: Semantic similarity cache (vector-based) - check cosine similarity > 0.95
    3. If cache miss → generate embedding
    4. Store in both caches for future use
    5. Optionally update vector DB with new embeddings (for feedback entries only)
    
    Args:
        texts: List of text strings to generate embeddings for
        use_smart_cache: If True, use text-hash-based caching (default: True)
        use_semantic_cache: If True, use semantic similarity cache (default: True)
        add_to_vector_db: If True, add new embeddings to vector DB (for feedback entries, default: False)
        vector_db_indices: List of indices for vector DB IDs (only used if add_to_vector_db=True)
        level_values: List of level values for metadata (only used if add_to_vector_db=True)
    
    Returns:
        numpy array of embeddings
    """
    cache = _load_embeddings_cache()
    model = _get_embedding_model()
    
    if use_smart_cache and len(texts) == 1:
        # Single text: use layered caching
        text = texts[0]
        # Normalize text and generate hash (hash is computed AFTER normalization)
        text_hash = _hash_text(text, normalized=True)
        cache_key_smart = f"text_{text_hash}"
        
        # Level 1: Check exact hash cache (fastest) - uses normalized text hash
        # If found, return immediately (already cached, no need to add to vector DB)
        if cache_key_smart in cache:
            cached_emb = cache[cache_key_smart]
            # Ensure it's a numpy array
            if isinstance(cached_emb, np.ndarray):
                return cached_emb
            else:
                return np.array(cached_emb)
        
        # Level 1 cache miss: Generate embedding (after normalized hashing)
        # Then check semantic cache and save to both caches
        embedding = None
        is_new_embedding = True  # Track if this is a new embedding (not found in any cache)
        
        # Level 2: Check semantic similarity cache (if enabled)
        if use_semantic_cache:
            semantic_cache = _load_semantic_cache()
            # Use same normalized hash as _add_to_semantic_cache uses
            semantic_key = f"semantic_{text_hash}"
            
            # Check for exact normalized text match first (no embedding generation needed)
            if semantic_key in semantic_cache:
                cached_entry = semantic_cache[semantic_key]
                if isinstance(cached_entry, dict) and 'embedding' in cached_entry:
                    # Exact match found - use cached embedding (no embedding generation needed!)
                    embedding = cached_entry['embedding']
                    is_new_embedding = False  # Found in cache, not new
                    # Also add to Level 1 cache for faster future access
                    cache[cache_key_smart] = embedding
                    _save_embeddings_cache()
                    # Update LRU
                    semantic_cache.move_to_end(semantic_key)
                    _save_semantic_cache()
                else:
                    # Generate embedding to check similarity
                    embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
                    similar_emb = _find_similar_in_semantic_cache(embedding, threshold=CACHE_SIMILARITY_THRESHOLD)
                    
                    if similar_emb is not None:
                        # Found similar embedding in cache - use it (not new)
                        embedding = similar_emb
                        is_new_embedding = False
                        # Also add to Level 1 cache for faster future access
                        cache[cache_key_smart] = embedding
                        _save_embeddings_cache()
            else:
                # No exact match: Generate embedding to check similarity
                embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
                similar_emb = _find_similar_in_semantic_cache(embedding, threshold=CACHE_SIMILARITY_THRESHOLD)
                
                if similar_emb is not None:
                    # Found similar embedding in cache - use it (not new)
                    embedding = similar_emb
                    is_new_embedding = False
                    # Also add to Level 1 cache for faster future access
                    cache[cache_key_smart] = embedding
                    _save_embeddings_cache()
                # No similar embedding found, use the one we just generated (is_new_embedding = True)
        else:
            # Generate embedding (semantic cache disabled)
            embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        
        # Save to both caches (either way - new or cached)
        # Level 1: Exact hash cache
        cache[cache_key_smart] = embedding
        _save_embeddings_cache()
        
        # Level 2: Semantic similarity cache
        if use_semantic_cache:
            _add_to_semantic_cache(text, embedding)
        
        # Update vector DB ONLY if this is a NEW embedding (not found in any cache)
        # AND it's a feedback entry (not ad-hoc query)
        if is_new_embedding and add_to_vector_db and vector_db_indices and len(vector_db_indices) == 1:
            level = level_values[0] if level_values and len(level_values) == 1 else None
            _add_embedding_to_vector_db(text, embedding, vector_db_indices[0], level=level)
        
        return embedding
    
    # Multiple texts: use batch processing with smart caching
    if use_smart_cache:
        # Check cache for each text individually
        embeddings_list = []
        texts_to_generate = []
        text_indices = []
        
        for i, text in enumerate(texts):
            text_hash = _hash_text(text)
            cache_key_smart = f"text_{text_hash}"
            
            if cache_key_smart in cache:
                embeddings_list.append((i, cache[cache_key_smart]))
            else:
                texts_to_generate.append(text)
                text_indices.append(i)
        
        # Generate embeddings for texts not in cache
        if texts_to_generate:
            new_embeddings = model.encode(texts_to_generate, show_progress_bar=False, convert_to_numpy=True)
            
            # Cache new embeddings
            for text, embedding in zip(texts_to_generate, new_embeddings):
                text_hash = _hash_text(text)
                cache_key_smart = f"text_{text_hash}"
                cache[cache_key_smart] = embedding
            _save_embeddings_cache()
            
            # Update vector DB with new embeddings if requested (for feedback entries only)
            # Only update when new embeddings are generated (not from cache)
            if add_to_vector_db and vector_db_indices:
                for i, (text, embedding) in enumerate(zip(texts_to_generate, new_embeddings)):
                    orig_text_idx = text_indices[i]
                    if orig_text_idx < len(vector_db_indices) and vector_db_indices[orig_text_idx] is not None:
                        db_idx = vector_db_indices[orig_text_idx]
                        level = level_values[orig_text_idx] if level_values and orig_text_idx < len(level_values) else None
                        _add_embedding_to_vector_db(text, embedding, db_idx, level=level)
            
            # Add new embeddings to list
            for idx, embedding in zip(text_indices, new_embeddings):
                embeddings_list.append((idx, embedding))
        
        # Reconstruct embeddings in original order
        embeddings_list.sort(key=lambda x: x[0])
        embeddings = np.array([emb[1] for emb in embeddings_list])
        
        return embeddings
    
    # Fallback: generate all at once (if smart cache disabled)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def _initialize_vector_db():
    """Initialize ChromaDB vector database for semantic search."""
    global _vector_db, _vector_collection
    
    if _vector_db is None:
        # Create persistent ChromaDB client
        db_path = os.path.join(os.getcwd(), '.chroma_db')
        os.makedirs(db_path, exist_ok=True)
        
        _vector_db = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection for feedback embeddings
        collection_name = "feedback_embeddings"
        try:
            _vector_collection = _vector_db.get_collection(name=collection_name)
        except:
            _vector_collection = _vector_db.create_collection(
                name=collection_name,
                metadata={"description": "Feedback text embeddings for semantic search"}
            )
    
    return _vector_collection

def _check_if_in_vector_db(index: int) -> bool:
    """
    Check if a feedback entry already exists in vector DB by ID.
    
    Args:
        index: Index in the feedback dataframe (position in non_empty list)
    
    Returns:
        True if exists, False otherwise
    """
    try:
        _vector_collection = _initialize_vector_db()
        existing_id = f"feedback_{index}"
        existing = _vector_collection.get(ids=[existing_id])
        return existing['ids'] and len(existing['ids']) > 0
    except:
        return False

def _check_if_hash_in_vector_db(text_hash: str) -> bool:
    """
    Check if a text hash already exists in vector DB.
    More reliable than ID-based checking - uses content hash.
    
    Args:
        text_hash: Normalized text hash to check
    
    Returns:
        True if hash exists in Vector DB, False otherwise
    """
    try:
        _vector_collection = _initialize_vector_db()
        # Query Vector DB by metadata text_hash
        results = _vector_collection.get(
            where={"text_hash": text_hash},
            limit=1
        )
        return results['ids'] and len(results['ids']) > 0
    except:
        return False

def _add_embedding_to_vector_db(text: str, embedding: np.ndarray, index: int, level: int = None, timestamp: str = None):
    """
    Add a single embedding to vector DB (for new feedback entries only).
    Only called when a NEW feedback entry is detected (not already in vector DB).
    This ensures vector DB stays updated with new arrivals efficiently.
    
    Args:
        text: Feedback text
        embedding: Embedding vector
        index: Index in the feedback dataframe (position in non_empty list)
        level: Optional level value for metadata
        timestamp: Optional timestamp value for metadata
    """
    try:
        _vector_collection = _initialize_vector_db()
        
        # Check if already exists (efficient detection of new arrivals)
        existing_id = f"feedback_{index}"
        if _check_if_in_vector_db(index):
            # Already exists, skip (not a new arrival)
            return
        
        # New arrival detected - add to vector DB
        text_hash = _hash_text(text, normalized=True)
        
        metadata = {
            "text": text[:500],  # Store first 500 chars
            "text_hash": text_hash
        }
        if level is not None:
            metadata["level"] = str(level)
        if timestamp is not None:
            metadata["timestamp"] = str(timestamp)
        
        # Add new entry to vector DB
        _vector_collection.add(
            embeddings=[embedding.tolist()],
            ids=[existing_id],
            metadatas=[metadata],
            documents=[text]
        )
    except Exception as e:
        # Silently fail - vector DB update is optional, cache is primary
        # This prevents errors from blocking the main flow
        pass

def _populate_vector_db():
    """
    Populate vector database with feedback embeddings.
    Uses smart caching: checks cache first to avoid recomputing embeddings.
    Only adds NEW entries (checks Vector DB for duplicates by text hash).
    """
    global feedback_df, _vector_collection
    
    if feedback_df is None:
        return
    
    _vector_collection = _initialize_vector_db()
    text_col = _get_text_column(feedback_df)
    if not text_col:
        return
    
    # Get non-empty text entries
    non_empty = feedback_df[feedback_df[text_col].notna() & (feedback_df[text_col].astype(str).str.strip() != '')].copy()
    non_empty = non_empty.reset_index(drop=True)
    total_feedback = len(non_empty)
    
    if total_feedback == 0:
        return
    
    # Find level and timestamp columns
    level_col = None
    timestamp_col = _get_timestamp_column(feedback_df)
    for col in feedback_df.columns:
        if col.lower() == 'level':
            level_col = col
            break
    
    # Get all existing text hashes from Vector DB in one batch query (FAST)
    existing_hashes = set()
    existing_count = _vector_collection.count()
    
    if existing_count > 0:
        try:
            # Get all existing entries with their metadata (one query)
            existing_results = _vector_collection.get(limit=existing_count, include=['metadatas'])
            if existing_results and existing_results.get('metadatas'):
                for metadata in existing_results['metadatas']:
                    if metadata and 'text_hash' in metadata:
                        existing_hashes.add(metadata['text_hash'])
        except:
            pass
    
    # Find new entries by checking text hash (fast in-memory lookup)
    new_entries = []
    new_indices = []
    
    for idx, row in non_empty.iterrows():
        text = str(row[text_col])
        text_hash = _hash_text(text, normalized=True)
        
        # Fast in-memory set lookup (O(1))
        if text_hash not in existing_hashes:
            new_entries.append(row)
            new_indices.append(idx)
    
    if len(new_entries) == 0:
        print(f"Vector database already up to date ({existing_count} entries)")
    elif existing_count == 0:
        # Empty DB - populate all entries
        print(f"Populating vector database with {total_feedback} feedback entries...")
        _add_entries_to_vector_db(non_empty, text_col, level_col, timestamp_col, batch_size=100)
        print(f"Vector database populated with {total_feedback} entries")
    else:
        # Some entries exist, add only new ones
        print(f"Updating vector database ({existing_count} existing, {total_feedback} total)...")
        print(f"   Found {len(new_entries)} new entries to add...")
        new_df = pd.DataFrame(new_entries).reset_index(drop=True)
        _add_entries_to_vector_db(new_df, text_col, level_col, timestamp_col, 
                                 start_index=min(new_indices) if new_indices else 0, 
                                 batch_size=100)
        print(f"Added {len(new_entries)} new entries to vector database")

def _add_entries_to_vector_db(non_empty: pd.DataFrame, text_col: str, level_col: Optional[str], 
                              timestamp_col: Optional[str], start_index: int = 0, batch_size: int = 100):
    """
    Add entries to Vector DB with caching and duplicate checking.
    
    Args:
        non_empty: DataFrame with non-empty text entries
        text_col: Text column name
        level_col: Level column name (optional)
        timestamp_col: Timestamp column name (optional)
        start_index: Starting index for Vector DB IDs
        batch_size: Batch size for processing
    """
    global _vector_collection
    
    texts = non_empty[text_col].astype(str).tolist()
    batch_indices_base = list(range(start_index, start_index + len(non_empty)))
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_indices = batch_indices_base[i:i+batch_size]
        batch_df_indices = list(range(i, min(i+batch_size, len(non_empty))))
        
        # Use _generate_embeddings() which handles all caching automatically (Level 1 + Level 2)
        # Caching happens FIRST - avoids regenerating embeddings
        embeddings = _generate_embeddings(batch_texts, use_smart_cache=True, use_semantic_cache=True)
        
        # Prepare metadata with text hash, level, and timestamp
        ids = [f"feedback_{idx}" for idx in batch_indices]
        metadatas = []
        for j, idx in enumerate(batch_df_indices):
            text = batch_texts[j]
            # Use normalized hash to match _populate_vector_db() checking logic
            text_hash = _hash_text(text, normalized=True)
            metadata = {
                "text": text[:500],  # Store first 500 chars
                "text_hash": text_hash  # Store hash for efficient lookups
            }
            if level_col and level_col in non_empty.columns:
                metadata["level"] = str(non_empty.iloc[idx][level_col])
            if timestamp_col and timestamp_col in non_empty.columns:
                timestamp_val = non_empty.iloc[idx][timestamp_col]
                if pd.notna(timestamp_val):
                    # Convert timestamp to string (handle various formats)
                    if isinstance(timestamp_val, pd.Timestamp):
                        metadata["timestamp"] = timestamp_val.isoformat()
                    else:
                        metadata["timestamp"] = str(timestamp_val)
            metadatas.append(metadata)
        
        # Check for duplicates before adding (by ID)
        ids_to_add = []
        embeddings_to_add = []
        metadatas_to_add = []
        documents_to_add = []
        
        for j, entry_id in enumerate(ids):
            # Check if already exists in Vector DB
            if not _check_if_in_vector_db(batch_indices[j]):
                ids_to_add.append(entry_id)
                # Handle both single embedding and batch embeddings
                if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:
                    embeddings_to_add.append(embeddings[j].tolist())
                elif isinstance(embeddings, list):
                    embeddings_to_add.append(embeddings[j].tolist() if hasattr(embeddings[j], 'tolist') else embeddings[j])
                else:
                    embeddings_to_add.append(embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings)
                metadatas_to_add.append(metadatas[j])
                documents_to_add.append(batch_texts[j])
        
        # Add only new entries (avoid duplicates)
        if ids_to_add:
            _vector_collection.add(
                embeddings=embeddings_to_add,
                ids=ids_to_add,
                metadatas=metadatas_to_add,
                documents=documents_to_add
            )

def _initialize_historical_db():
    """
    Initialize SQLite database for long-term historical data storage.
    This stores all feedback entries with timestamps for trend analysis over time.
    """
    global _historical_db_path
    
    if _historical_db_path is None:
        db_dir = os.path.join(os.getcwd(), '.historical_db')
        os.makedirs(db_dir, exist_ok=True)
        _historical_db_path = os.path.join(db_dir, 'feedback_history.db')
        
        # Create table if it doesn't exist
        conn = sqlite3.connect(_historical_db_path)
        cursor = conn.cursor()
        
        # Create table with flexible schema (store all columns as JSON)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE,
                stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_data TEXT,
                level INTEGER,
                timestamp TEXT
            )
        ''')
        
        # Create indexes for efficient querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_hash ON feedback_history(text_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stored_at ON feedback_history(stored_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback_history(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON feedback_history(level)')
        
        conn.commit()
        conn.close()
    
    return _historical_db_path

def _store_to_historical_db(dataframe: pd.DataFrame):
    """
    Store current feedback data to historical database (long-term memory).
    Only stores NEW entries (checks for duplicates by text hash).
    Efficient: Uses caching and duplicate detection.
    
    Args:
        dataframe: Current feedback dataframe to store
    """
    global feedback_df
    
    if dataframe is None or len(dataframe) == 0:
        return
    
    try:
        db_path = _initialize_historical_db()
        text_col = _get_text_column(dataframe)
        if not text_col:
            return
        
        # Find level and timestamp columns
        level_col = None
        timestamp_col = _get_timestamp_column(dataframe)
        for col in dataframe.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get non-empty text entries
        non_empty = dataframe[dataframe[text_col].notna() & (dataframe[text_col].astype(str).str.strip() != '')].copy()
        
        new_count = 0
        duplicate_count = 0
        
        for idx, row in non_empty.iterrows():
            text = str(row[text_col])
            text_hash = _hash_text(text, normalized=True)
            
            # Check if already exists (efficient duplicate detection)
            cursor.execute('SELECT id FROM feedback_history WHERE text_hash = ?', (text_hash,))
            if cursor.fetchone():
                duplicate_count += 1
                continue  # Skip duplicate
            
            # Prepare data: store all columns as JSON
            feedback_data = {}
            for col in dataframe.columns:
                val = row[col]
                if pd.notna(val):
                    # Convert to JSON-serializable format
                    if isinstance(val, (pd.Timestamp, datetime)):
                        feedback_data[col] = val.isoformat()
                    else:
                        feedback_data[col] = str(val)
            
            import json
            feedback_json = json.dumps(feedback_data)
            
            # Get level and timestamp for indexing
            level_val = None
            if level_col and level_col in row:
                try:
                    level_val = int(row[level_col]) if pd.notna(row[level_col]) else None
                except:
                    level_val = None
            
            timestamp_val = None
            if timestamp_col and timestamp_col in row:
                timestamp_val = str(row[timestamp_col]) if pd.notna(row[timestamp_col]) else None
            
            # Insert new entry
            cursor.execute('''
                INSERT INTO feedback_history (text_hash, feedback_data, level, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (text_hash, feedback_json, level_val, timestamp_val))
            
            new_count += 1
        
        conn.commit()
        conn.close()
        
        if new_count > 0:
            print(f"Stored {new_count} new entries to historical database (long-term memory)")
            if duplicate_count > 0:
                print(f"   Skipped {duplicate_count} duplicates (already in historical DB)")
    except Exception as e:
        # Silently fail - historical storage is optional
        pass

def _get_historical_data(start_date: str = None, end_date: str = None, 
                         level_filter: int = None, limit: int = None) -> pd.DataFrame:
    """
    Retrieve historical feedback data from long-term memory.
    
    Args:
        start_date: Start date filter (YYYY-MM-DD, optional)
        end_date: End date filter (YYYY-MM-DD, optional)
        level_filter: Filter by level (optional)
        limit: Maximum number of records to return (optional)
    
    Returns:
        DataFrame with historical feedback data
    """
    try:
        db_path = _initialize_historical_db()
        conn = sqlite3.connect(db_path)
        
        query = 'SELECT feedback_data, level, timestamp, stored_at FROM feedback_history WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND (timestamp >= ? OR stored_at >= ?)'
            params.extend([start_date, start_date])
        
        if end_date:
            query += ' AND (timestamp <= ? OR stored_at <= ?)'
            params.extend([end_date, end_date])
        
        if level_filter is not None:
            query += ' AND level = ?'
            params.append(level_filter)
        
        query += ' ORDER BY stored_at DESC'
        
        if limit:
            query += f' LIMIT {limit}'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Parse JSON data back to columns
        import json
        rows = []
        for _, row in df.iterrows():
            try:
                data = json.loads(row['feedback_data'])
                data['_historical_stored_at'] = row['stored_at']
                rows.append(data)
            except:
                continue
        
        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def initialize_tools(dataframe: pd.DataFrame):
    """
    Initialize tools with the feedback dataframe.
    This handles BOTH short-term (current CSV) and long-term (historical DB) memory.
    
    Args:
        dataframe: The pandas DataFrame containing feedback data
    """
    global feedback_df
    feedback_df = dataframe.copy()
    
    # Pre-load embedding model (lazy loading, but this ensures it's ready)
    _get_embedding_model()
    
    # SHORT-TERM MEMORY: Initialize and populate vector database (for immediate queries)
    _populate_vector_db()
    
    # LONG-TERM MEMORY: Store to historical database (for trend analysis over time)
    # Only stores NEW entries (efficient duplicate detection)
    _store_to_historical_db(dataframe)


@tool
def get_feedback_by_level(level: int) -> str:
    """
    Get all feedback entries with a specific level (ranking).
    
    Args:
        level: The numeric level/ranking to filter by
        
    Returns:
        A string representation of matching feedback entries
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        # Find level column (case-insensitive)
        level_col = None
        for col in feedback_df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        if level_col is None:
            return "Error: 'level' column not found in dataframe"
        
        filtered = feedback_df[feedback_df[level_col] == level]
        if len(filtered) == 0:
            return f"No feedback entries found with level {level}"
        
        result = f"Found {len(filtered)} feedback entries with level {level}:\n\n"
        # Format text column for RTL display
        text_col = _get_text_column(filtered)
        display_df = filtered.copy()
        if text_col and text_col in display_df.columns:
            display_df[text_col] = display_df[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
        # Show limited rows to avoid overwhelming output
        if len(display_df) > 10:
            result += display_df.head(10).to_string()
            result += f"\n\n... and {len(display_df) - 10} more entries (showing first 10)"
        else:
            result += display_df.to_string()
        return result
    except Exception as e:
        return f"Error filtering by level: {str(e)}"


@tool
def analyze_feedback_themes(
    use_clustering: bool = False,
    level_filter: int = None,
    min_level: int = None,
    max_level: int = None,
    filter_by_level: bool = False,
    min_cluster_size: int = 3,
    eps: float = 0.4
) -> str:
    """
    Hybrid function to analyze feedback themes using either clustering (automatic grouping) or LLM analysis (action-oriented).
    Level filtering is applied ONLY if filter_by_level=True (based on user intent).
    
    Args:
        use_clustering: If True, use DBSCAN clustering for automatic grouping. If False, use LLM-based analysis.
        level_filter: Specific level to filter by (only applied if filter_by_level=True)
        min_level: Minimum level threshold (only applied if filter_by_level=True)
        max_level: Maximum level threshold (only applied if filter_by_level=True)
        filter_by_level: If True, apply level filtering. If False, analyze ALL feedback regardless of level.
        min_cluster_size: Minimum cluster size for DBSCAN (default: 3)
        eps: Maximum distance for DBSCAN clustering (default: 0.4, lower = stricter)
        
    Returns:
        Analysis with themes, categories, statistics, and action items (or clusters if use_clustering=True)
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        text_col = _get_text_column(feedback_df)
        if not text_col:
            return "Error: No text/feedback column found in dataframe"
        
        # Find level column
        level_col = None
        for col in feedback_df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        # Apply level filtering ONLY if user explicitly wants it (filter_by_level=True)
        filtered_df = feedback_df.copy()
        if filter_by_level and level_col:
            if level_filter is not None:
                filtered_df = filtered_df[filtered_df[level_col] == level_filter]
            elif min_level is not None:
                filtered_df = filtered_df[filtered_df[level_col] <= min_level]
            elif max_level is not None:
                filtered_df = filtered_df[filtered_df[level_col] >= max_level]
        
        # Get non-empty text entries
        non_empty = filtered_df[filtered_df[text_col].notna() & (filtered_df[text_col].astype(str).str.strip() != '')].copy()
        
        if len(non_empty) == 0:
            return "No feedback text found matching the criteria."
        
        # Choose method: Clustering or LLM Analysis
        if use_clustering:
            # CLUSTERING MODE: Automatic grouping using DBSCAN with vector DB
            if len(non_empty) < min_cluster_size:
                return f"Not enough feedback entries ({len(non_empty)}) for clustering. Need at least {min_cluster_size} entries."
            
            # Use vector DB to get embeddings (much faster than generating on the fly)
            _vector_collection = _initialize_vector_db()
            
            # Vector DB stores entries in order: feedback_0, feedback_1, ... based on position in non_empty dataframe
            # We need to map filtered entries back to their positions in the full non_empty dataframe
            # Get all non-empty entries from original dataframe (same way as when populating)
            all_non_empty = feedback_df[feedback_df[text_col].notna() & (feedback_df[text_col].astype(str).str.strip() != '')].copy()
            
            # Create mapping: original dataframe index -> position in all_non_empty (0-based)
            # The vector DB IDs are feedback_0, feedback_1, ... where the number is the position in all_non_empty
            # Before reset_index, preserve the mapping
            index_to_position = {}
            for pos, orig_idx in enumerate(all_non_empty.index):
                index_to_position[orig_idx] = pos
            
            # Get positions of filtered entries in the all_non_empty dataframe
            vector_ids = []
            for orig_idx in non_empty.index:
                if orig_idx in index_to_position:
                    pos = index_to_position[orig_idx]
                    vector_ids.append(f"feedback_{pos}")
            
            # Get embeddings: Cache first (fastest), then generate if needed
            texts = non_empty[text_col].astype(str).tolist()
            
            # Get vector DB indices for updating vector DB with new embeddings (only when necessary)
            all_non_empty = feedback_df[feedback_df[text_col].notna() & (feedback_df[text_col].astype(str).str.strip() != '')].copy()
            all_non_empty = all_non_empty.reset_index(drop=True)
            index_to_position = {}
            for pos, orig_idx in enumerate(all_non_empty.index):
                index_to_position[orig_idx] = pos
            
            vector_db_indices = []
            for orig_idx in non_empty.index:
                if orig_idx in index_to_position:
                    vector_db_indices.append(index_to_position[orig_idx])
                else:
                    vector_db_indices.append(None)
            
            # Get level values for metadata (if available)
            level_col = None
            for col in feedback_df.columns:
                if col.lower() == 'level':
                    level_col = col
                    break
            
            level_values = []
            for orig_idx in non_empty.index:
                if level_col and orig_idx in non_empty.index:
                    level_values.append(non_empty.loc[orig_idx, level_col])
                else:
                    level_values.append(None)
            
            # Try cache first for maximum efficiency (Level 1 + Level 2)
            # _generate_embeddings() will check cache first, then generate if needed
            # New embeddings will be automatically added to vector DB (only when necessary)
            embeddings = _generate_embeddings(
                texts, 
                use_smart_cache=True, 
                use_semantic_cache=True,
                add_to_vector_db=True,
                vector_db_indices=vector_db_indices,
                level_values=level_values
            )
            
            # Note: Cache is checked first (fastest), vector DB is updated only for new embeddings
            
            # Perform clustering
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine')
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Add cluster labels
            non_empty['cluster'] = cluster_labels
            
            # Analyze clusters
            unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]  # -1 is noise
            noise_count = len(non_empty[non_empty['cluster'] == -1])
            
            result = f"=== FEEDBACK CLUSTERING RESULTS ===\n\n"
            result += f"Total entries analyzed: {len(non_empty)}\n"
            if filter_by_level and level_col:
                if level_filter:
                    result += f"Level filter: {level_filter}\n"
                elif min_level:
                    result += f"Level threshold: <= {min_level}\n"
                elif max_level:
                    result += f"Level threshold: >= {max_level}\n"
            else:
                result += f"Analyzing ALL levels (semantic analysis regardless of rating)\n"
            result += f"Number of clusters found: {len(unique_clusters)}\n"
            result += f"Noise/Outliers (not in any cluster): {noise_count}\n"
            result += f"Clustering parameters: eps={eps}, min_cluster_size={min_cluster_size}\n\n"
            
            # Show each cluster
            for cluster_id in sorted(unique_clusters):
                cluster_data = non_empty[non_empty['cluster'] == cluster_id]
                result += f"--- Cluster {cluster_id} ({len(cluster_data)} entries) ---\n"
                
                # Show sample entries
                display_cols = [text_col] + ([level_col] if level_col else [])
                sample_size = min(5, len(cluster_data))
                result += f"Sample entries:\n"
                # Format text column for RTL display
                cluster_display = cluster_data.head(sample_size)[display_cols].copy()
                if text_col in cluster_display.columns:
                    cluster_display[text_col] = cluster_display[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
                result += cluster_display.to_string()
                result += f"\n\n"
            
            if noise_count > 0:
                result += f"--- Noise/Outliers ({noise_count} entries not in any cluster) ---\n"
                noise_sample = non_empty[non_empty['cluster'] == -1].head(3)
                if len(noise_sample) > 0:
                    # Format text column for RTL display
                    noise_display = noise_sample[[text_col] + ([level_col] if level_col else [])].copy()
                    if text_col in noise_display.columns:
                        noise_display[text_col] = noise_display[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
                    result += noise_display.to_string()
                    result += f"\n\n"
            
            result += f"\n⚠️ INSTRUCTIONS FOR LLM:\n"
            result += f"Analyze each cluster to:\n"
            result += f"- Identify the common theme/topic for each cluster\n"
            result += f"- Name each cluster based on its content\n"
            result += f"- Provide statistics: How many entries in each cluster\n"
            result += f"- Extract key insights from each cluster\n"
            result += f"- Note: Clustering is based on semantic similarity using embeddings\n"
            
        else:
            # LLM ANALYSIS MODE: Action-oriented analysis with categorization
            result = f"=== FEEDBACK THEME ANALYSIS ===\n\n"
            result += f"Total entries to analyze: {len(non_empty)}\n"
            
            if filter_by_level and level_col:
                if level_filter:
                    result += f"Level filter: {level_filter}\n"
                elif min_level:
                    result += f"Level threshold: <= {min_level}\n"
                elif max_level:
                    result += f"Level threshold: >= {max_level}\n"
            else:
                result += f"Analyzing ALL feedback entries semantically, regardless of level rating.\n"
                result += f"This ensures we catch improvement suggestions even in high-rated feedback.\n"
            
            result += f"\n⚠️ CRITICAL INSTRUCTIONS FOR LLM:\n"
            result += f"1. Perform SEMANTIC analysis of text content, NOT just level rating\n"
            result += f"2. Identify complaints/issues even in high-level feedback (level 4-5) if analyzing all levels\n"
            result += f"3. Categorize issues/themes and PROVIDE STATISTICS:\n"
            result += f"   - Count how many times each complaint/theme appears\n"
            result += f"   - Show format: 'Category Name: X occurrences (Y%)'\n"
            result += f"4. Group similar complaints together\n"
            result += f"5. Provide action items for each category:\n"
            result += f"   - For negative feedback: Improvement suggestions\n"
            result += f"   - For positive feedback: What to keep/maintain\n"
            result += f"6. Provide key takeaways with occurrence counts\n"
            
            result += f"\nSample feedback entries for analysis:\n"
            display_cols = [text_col] + ([level_col] if level_col else [])
            # Format text column for RTL display
            display_df = non_empty.head(30)[display_cols].copy()
            if text_col in display_df.columns:
                display_df[text_col] = display_df[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
            result += display_df.to_string()
            
            if len(non_empty) > 30:
                result += f"\n\n... and {len(non_empty) - 30} more entries\n"
        
        return result
    except Exception as e:
        return f"Error analyzing feedback themes: {str(e)}"

@tool
def generate_categorized_issues_report(level_filter: int = None, min_level: int = None, max_level: int = None, categories: list = None) -> str:
    """
    Generate a CSV report with categories as columns. Each category becomes a column (1 if the feedback 
    relates to that category, 0 otherwise). Works for both positive and negative feedback.
    
    Args:
        level_filter: Specific level to analyze (e.g., 1, 2, 3 for negative; 4, 5 for positive)
        min_level: Minimum level threshold (e.g., analyze all feedback with level <= 3 for negative)
        max_level: Maximum level threshold (e.g., analyze all feedback with level >= 4 for positive)
        categories: List of category names to use as columns. 
                    Default: ["Usability", "Performance", "Support", "Features", "Errors", "Documentation", "Accessibility"]
    
    Returns:
        A message indicating the CSV file was created with category columns and occurrence counts
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        import os
        from datetime import datetime
        
        # Default categories if not provided
        if categories is None:
            categories = ["Usability", "Performance", "Support", "Features", "Errors", "Documentation", "Accessibility"]
        
        # Find text and level columns
        text_col = _get_text_column(feedback_df)
        if not text_col:
            return "Error: No 'text' column found in dataframe"
        
        level_col = None
        for col in feedback_df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        # Filter by level if specified
        filtered_df = feedback_df.copy()
        if level_filter is not None:
            if level_col:
                filtered_df = filtered_df[filtered_df[level_col] == level_filter]
        elif min_level is not None:
            if level_col:
                filtered_df = filtered_df[filtered_df[level_col] <= min_level]
        elif max_level is not None:
            if level_col:
                filtered_df = filtered_df[filtered_df[level_col] >= max_level]
        
        # Get non-empty text entries
        non_empty = filtered_df[filtered_df[text_col].notna() & (filtered_df[text_col].astype(str).str.strip() != '')]
        
        if len(non_empty) == 0:
            return "No feedback text found matching the criteria."
        
        # Create report dataframe with all original columns
        report_df = non_empty.copy()
        
        # Initialize category columns with 0 (will be filled by LLM analysis)
        # The LLM should analyze each feedback entry and set appropriate category columns to 1
        for category in categories:
            report_df[f'Category_{category}'] = 0
        
        # Add a note column for the LLM to add categorization notes if needed
        report_df['Categorization_Notes'] = ''
        
        # Generate filename
        filter_desc = ""
        if level_filter is not None:
            filter_desc = f"Level_{level_filter}"
        elif min_level is not None:
            filter_desc = f"LevelMax_{min_level}"
        elif max_level is not None:
            filter_desc = f"LevelMin_{max_level}"
        else:
            filter_desc = "All"
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_filename = f"feedback_issues_categorized_{filter_desc}_{date_str}.csv"
        
        # Calculate category occurrence counts (will be 0 initially, but structure is ready)
        category_counts = {}
        for category in categories:
            category_counts[category] = int(report_df[f'Category_{category}'].sum())
        
        # Create summary row with category counts
        summary_row = {}
        # Fill with empty strings for original columns
        for col in report_df.columns:
            if col.startswith('Category_') or col == 'Categorization_Notes':
                summary_row[col] = ''
            else:
                summary_row[col] = ''
        
        # Add summary label
        summary_row[report_df.columns[0]] = 'SUMMARY_TOTAL_COUNTS'
        
        # Add category counts to summary row
        for category in categories:
            summary_row[f'Category_{category}'] = category_counts[category]
        
        # Add summary row to dataframe
        summary_df = pd.DataFrame([summary_row])
        report_df_with_summary = pd.concat([report_df, summary_df], ignore_index=True)
        
        # Save to CSV
        output_path = os.path.join(os.getcwd(), output_filename)
        report_df_with_summary.to_csv(output_path, index=False, encoding='utf-8')
        
        # Build result message with category counts and statistics
        result = f"✅ Categorized feedback report generated!\n"
        result += f"   File: {output_filename}\n"
        result += f"   Rows: {len(report_df)} feedback entries\n"
        result += f"\n   Category Statistics (Occurrence Counts):\n"
        total_with_categories = sum(1 for cat in categories if category_counts[cat] > 0)
        for category in categories:
            count = category_counts[category]
            percentage = (count / len(report_df) * 100) if len(report_df) > 0 else 0
            result += f"   - {category}: {count} occurrences ({percentage:.1f}%)\n"
        result += f"\n   Total entries with at least one category: {total_with_categories}\n"
        result += f"   Category columns: {', '.join([f'Category_{cat}' for cat in categories])}\n"
        result += f"   Note: Category columns are initialized to 0. The LLM should analyze each feedback\n"
        result += f"         entry and set appropriate category columns to 1 based on semantic content,\n"
        result += f"         regardless of the level rating. The last row contains summary counts.\n"
        result += f"   Location: {output_path}"
        
        return result
    except Exception as e:
        return f"Error generating categorized report: {str(e)}"

@tool
def get_feedback_statistics() -> str:
    """
    Get comprehensive statistics about ALL columns in the feedback data.
    Provides statistics for numeric columns, categorical columns, text columns, and timestamps.
    
    Returns:
        Comprehensive statistics about the feedback dataset including all columns
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        result = "=== FEEDBACK STATISTICS ===\n\n"
        result += f"Total feedback entries: {len(feedback_df)}\n"
        result += f"Total columns: {len(feedback_df.columns)}\n"
        result += f"Columns: {', '.join(feedback_df.columns.tolist())}\n\n"
        
        # Level statistics if level column exists (case-insensitive)
        level_col = None
        for col in feedback_df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        if level_col:
            result += "=== LEVEL DISTRIBUTION ===\n"
            level_counts = feedback_df[level_col].value_counts().sort_index()
            result += level_counts.to_string()
            result += f"\n\nAverage level: {feedback_df[level_col].mean():.2f}\n"
            result += f"Min level: {feedback_df[level_col].min()}\n"
            result += f"Max level: {feedback_df[level_col].max()}\n\n"
        
        # Text column statistics
        text_col = _get_text_column(feedback_df)
        if text_col:
            result += f"=== TEXT FEEDBACK ANALYSIS ===\n"
            result += f"Column: {text_col}\n"
            result += f"Non-empty entries: {feedback_df[text_col].notna().sum()}\n"
            result += f"Average text length: {feedback_df[text_col].astype(str).str.len().mean():.1f} characters\n\n"
        
        # Timestamp column statistics
        timestamp_col = _get_timestamp_column(feedback_df)
        if timestamp_col:
            try:
                df_with_dates = feedback_df.copy()
                df_with_dates[timestamp_col] = pd.to_datetime(df_with_dates[timestamp_col], errors='coerce')
                valid_dates = df_with_dates[timestamp_col].notna()
                if valid_dates.sum() > 0:
                    result += f"=== TIMESTAMP ANALYSIS ===\n"
                    result += f"Column: {timestamp_col}\n"
                    result += f"Date range: {df_with_dates[timestamp_col].min()} to {df_with_dates[timestamp_col].max()}\n"
                    result += f"Valid timestamps: {valid_dates.sum()} ({valid_dates.sum()/len(feedback_df)*100:.1f}%)\n\n"
            except:
                pass
        
        # Statistics for all other columns
        result += "=== ALL COLUMNS SUMMARY ===\n"
        for col in feedback_df.columns:
            if col == text_col or col == level_col or col == timestamp_col:
                continue  # Already covered above
            
            result += f"\n{col}:\n"
            result += f"  Type: {feedback_df[col].dtype}\n"
            result += f"  Non-null: {feedback_df[col].notna().sum()} ({feedback_df[col].notna().sum()/len(feedback_df)*100:.1f}%)\n"
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(feedback_df[col]):
                result += f"  Min: {feedback_df[col].min()}\n"
                result += f"  Max: {feedback_df[col].max()}\n"
                result += f"  Mean: {feedback_df[col].mean():.2f}\n"
                result += f"  Unique values: {feedback_df[col].nunique()}\n"
            
            # Categorical/string columns
            elif pd.api.types.is_object_dtype(feedback_df[col]) or pd.api.types.is_string_dtype(feedback_df[col]):
                unique_count = feedback_df[col].nunique()
                result += f"  Unique values: {unique_count}\n"
                if unique_count <= 20:
                    # Show value counts for columns with few unique values
                    value_counts = feedback_df[col].value_counts().head(10)
                    result += f"  Top values:\n"
                    for val, count in value_counts.items():
                        result += f"    '{val}': {count} ({count/len(feedback_df)*100:.1f}%)\n"
                else:
                    # Just show sample values
                    sample_values = feedback_df[col].dropna().unique()[:5]
                    result += f"  Sample values: {', '.join([str(v) for v in sample_values])}\n"
        
        return result
    except Exception as e:
        return f"Error getting statistics: {str(e)}"

@tool
def generate_dataframe_report(filter_criteria: dict, columns: list = None, output_filename: str = None) -> str:
    """
    Generate a dataframe report with filtered data and specified columns, saved as a CSV file.
    This tool creates a filtered dataframe based on criteria and saves it to a CSV file.
    
    Args:
        filter_criteria: Dictionary with column names as keys and filter values. 
                        Examples: {"Level": 5}, {"Level": [4, 5]}, {"Text": "service"}
        columns: List of column names to include in the report. If None, includes all columns.
                 Example: ["ID", "Level", "Text", "ReferenceNumber"]
        output_filename: Optional filename for the CSV. If None, generates a descriptive filename.
    
    Returns:
        A message indicating the CSV file was created with the file path
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        import os
        from datetime import datetime
        
        filtered_df = feedback_df.copy()
        
        # Apply filters
        for col_name, filter_value in filter_criteria.items():
            # Find column (case-insensitive)
            col = None
            for c in filtered_df.columns:
                if c.lower() == col_name.lower():
                    col = c
                    break
            
            if col is None:
                return f"Error: Column '{col_name}' not found in dataframe"
            
            if isinstance(filter_value, list):
                # Multiple values (OR condition)
                filtered_df = filtered_df[filtered_df[col].isin(filter_value)]
            elif isinstance(filter_value, str):
                # String search (contains)
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(filter_value, case=False, na=False)]
            else:
                # Exact match
                filtered_df = filtered_df[filtered_df[col] == filter_value]
        
        # Select columns
        if columns:
            # Find columns (case-insensitive)
            selected_cols = []
            for col_name in columns:
                found = False
                for c in filtered_df.columns:
                    if c.lower() == col_name.lower():
                        selected_cols.append(c)
                        found = True
                        break
                if not found:
                    return f"Error: Column '{col_name}' not found in dataframe"
            filtered_df = filtered_df[selected_cols]
        
        if len(filtered_df) == 0:
            return "No data matches the filter criteria."
        
        # Generate filename if not provided
        if output_filename is None:
            # Create descriptive filename with date
            filter_desc = "_".join([f"{k}_{v}" for k, v in filter_criteria.items()])
            filter_desc = filter_desc.replace(" ", "_").replace("[", "").replace("]", "").replace(",", "_")
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_filename = f"feedback_report_{filter_desc}_{date_str}.csv"
        
        # Ensure .csv extension
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        # Save to CSV
        output_path = os.path.join(os.getcwd(), output_filename)
        filtered_df.to_csv(output_path, index=False, encoding='utf-8')
        
        result = f"✅ CSV report generated successfully!\n"
        result += f"   File: {output_filename}\n"
        result += f"   Rows: {len(filtered_df)}\n"
        result += f"   Columns: {', '.join(filtered_df.columns.tolist())}\n"
        result += f"   Location: {output_path}"
        
        return result
    except Exception as e:
        return f"Error generating report: {str(e)}"

@tool
def semantic_search_feedback(
    query: str = None,
    reference_text: str = None,
    reference_index: int = None,
    top_k: int = 10,
    level_filter: int = None,
    min_level: int = None,
    max_level: int = None,
    filter_by_level: bool = False,
    similarity_threshold: float = 0.3
) -> str:
    """
    Unified semantic search tool using vector DB. Can search by query string OR find entries similar to a reference.
    Uses the CachedFeedbackRetriever internally for query string searches (with all caching benefits).
    Handles reference modes separately (reference_text, reference_index) for advanced features.
    
    Args:
        query: Search query string (e.g., "file upload problems", "customer service issues")
        reference_text: Text to find similar entries for (alternative to query)
        reference_index: Index of a feedback entry in the dataframe (0-based, alternative to query)
        top_k: Number of most similar results to return (default: 10)
        level_filter: Filter by specific level (only applied if filter_by_level=True)
        min_level: Filter by minimum level threshold (only applied if filter_by_level=True)
        max_level: Filter by maximum level threshold (only applied if filter_by_level=True)
        filter_by_level: If True, apply level filtering. If False, search ALL feedback regardless of level.
        similarity_threshold: Minimum similarity score (0-1) to include results (default: 0.3)
        
    Returns:
        List of most similar feedback entries with similarity scores
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        text_col = _get_text_column(feedback_df)
        if not text_col:
            return "Error: No text/feedback column found in dataframe"
        
        # Find level column
        level_col = None
        for col in feedback_df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        # Get non-empty text entries for reference lookup
        non_empty = feedback_df[feedback_df[text_col].notna() & (feedback_df[text_col].astype(str).str.strip() != '')].copy()
        non_empty = non_empty.reset_index(drop=True)
        
        if len(non_empty) == 0:
            return "No feedback text found in dataframe."
        
        # Handle reference modes (reference_text, reference_index) - convert to query string
        actual_query = query
        reference_idx = None
        reference_text_used = None
        
        if reference_index is not None:
            # Mode: Find similar to reference index
            if reference_index < 0 or reference_index >= len(non_empty):
                return f"Error: Reference index {reference_index} is out of range. Valid range: 0-{len(non_empty)-1}"
            reference_text_used = non_empty.iloc[reference_index][text_col]
            reference_idx = reference_index
            # Convert to query string to use retriever
            actual_query = reference_text_used
        elif reference_text:
            # Mode: Find similar to reference text (find closest match first)
            ref_emb = _generate_embeddings([reference_text], use_smart_cache=True, use_semantic_cache=True)
            _vector_collection = _initialize_vector_db()
            query_results = _vector_collection.query(
                query_embeddings=ref_emb.tolist(),
                n_results=1
            )
            if query_results['ids'] and len(query_results['ids'][0]) > 0:
                reference_idx = int(query_results['ids'][0][0].split('_')[1])
                reference_text_used = non_empty.iloc[reference_idx][text_col]
                # Convert to query string to use retriever
                actual_query = reference_text_used
            else:
                return "Error: Could not find reference text in vector database"
        elif query:
            # Mode: Query string search - use retriever directly
            actual_query = query
        else:
            return "Error: Must provide either query, reference_text, or reference_index"
        
        # Use retriever for query string searches (with all caching benefits!)
        from retriever import CachedFeedbackRetriever
        retriever = CachedFeedbackRetriever(
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Get documents from retriever (uses RAG cache, embedding cache, etc.)
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        documents = retriever._get_relevant_documents(
            actual_query,
            run_manager=CallbackManagerForRetrieverRun.get_noop_manager()
        )
        
        # Filter by level if needed (retriever doesn't support level filtering yet)
        if filter_by_level and level_col:
            filtered_documents = []
            for doc in documents:
                doc_level = doc.metadata.get('level')
                if doc_level is not None:
                    doc_level = int(doc_level)
                    if level_filter is not None and doc_level != level_filter:
                        continue
                    if min_level is not None and doc_level > min_level:
                        continue
                    if max_level is not None and doc_level < max_level:
                        continue
                filtered_documents.append(doc)
            documents = filtered_documents[:top_k]
        
        # Exclude reference entry itself if searching by reference
        if reference_idx is not None:
            documents = [doc for doc in documents if doc.metadata.get('df_index') != reference_idx]
            documents = documents[:top_k]
        
        if len(documents) == 0:
            return f"No feedback found with similarity >= {similarity_threshold}"
        
        # Build result message
        result = f"=== SEMANTIC SEARCH RESULTS (Vector DB) ===\n\n"
        if reference_idx is not None:
            result += f"Reference entry (index {reference_idx}):\n"
            if level_col:
                result += f"  Level: {non_empty.iloc[reference_idx][level_col]}\n"
            result += f"  Text: {_format_text_for_display(reference_text_used, max_length=300)}\n\n"
            result += f"Searching for: entries similar to reference\n"
        else:
            result += f"Query: '{query}'\n"
        result += f"Found {len(documents)} similar feedback entries (similarity >= {similarity_threshold})\n"
        if filter_by_level and level_col:
            if level_filter:
                result += f"Level filter: {level_filter}\n"
            elif min_level:
                result += f"Level threshold: <= {min_level}\n"
            elif max_level:
                result += f"Level threshold: >= {max_level}\n"
        result += f"\n"
        
        for idx, doc in enumerate(documents, 1):
            result += f"Result {idx} (Similarity: {doc.metadata.get('similarity', 0):.3f}):\n"
            if 'level' in doc.metadata:
                result += f"  Level: {doc.metadata['level']}\n"
            result += f"  Text: {_format_text_for_display(doc.page_content, max_length=200)}\n\n"
        
        result += f"\n⚠️ INSTRUCTIONS FOR LLM:\n"
        result += f"Analyze these semantically similar feedback entries to:\n"
        result += f"- Identify common themes and patterns\n"
        result += f"- Extract key insights from the similar entries\n"
        result += f"- Provide statistics on what these entries reveal\n"
        result += f"- Note: These results are based on semantic similarity using vector DB\n"
        result += f"- Note: Results use the CachedFeedbackRetriever with RAG cache and embedding cache\n"
        
        return result
    except Exception as e:
        return f"Error performing semantic search: {str(e)}"



@tool
def get_historical_feedback(start_date: str = None, end_date: str = None, 
                             level_filter: int = None, limit: int = 1000) -> str:
    """
    Retrieve historical feedback data from long-term memory (historical database).
    Use this to analyze trends over time or compare current data with historical data.
    
    Args:
        start_date: Start date filter (YYYY-MM-DD format, optional)
        end_date: End date filter (YYYY-MM-DD format, optional)
        level_filter: Filter by specific level (optional)
        limit: Maximum number of records to return (default: 1000)
    
    Returns:
        Summary of historical feedback data with statistics
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        historical_df = _get_historical_data(start_date, end_date, level_filter, limit)
        
        if len(historical_df) == 0:
            date_range = ""
            if start_date or end_date:
                date_range = f" in date range {start_date or 'beginning'} to {end_date or 'end'}"
            return f"No historical feedback found{date_range}."
        
        result = f"=== HISTORICAL FEEDBACK DATA (Long-Term Memory) ===\n\n"
        result += f"Total historical entries: {len(historical_df)}\n"
        
        if start_date or end_date:
            result += f"Date range: {start_date or 'beginning'} to {end_date or 'end'}\n"
        
        if level_filter is not None:
            result += f"Level filter: {level_filter}\n"
        
        result += f"\nColumns available: {', '.join(historical_df.columns.tolist())}\n\n"
        
        # Level distribution if level column exists
        level_col = None
        for col in historical_df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        if level_col and level_col in historical_df.columns:
            result += "=== LEVEL DISTRIBUTION (Historical) ===\n"
            level_counts = historical_df[level_col].value_counts().sort_index()
            result += level_counts.to_string()
            result += f"\n\nAverage level: {historical_df[level_col].mean():.2f}\n\n"
        
        # Timestamp analysis
        timestamp_col = _get_timestamp_column(historical_df)
        if timestamp_col and timestamp_col in historical_df.columns:
            try:
                df_with_dates = historical_df.copy()
                df_with_dates[timestamp_col] = pd.to_datetime(df_with_dates[timestamp_col], errors='coerce')
                valid_dates = df_with_dates[timestamp_col].notna()
                if valid_dates.sum() > 0:
                    result += f"=== TIMESTAMP RANGE (Historical) ===\n"
                    result += f"Earliest: {df_with_dates[timestamp_col].min()}\n"
                    result += f"Latest: {df_with_dates[timestamp_col].max()}\n\n"
            except:
                pass
        
        # Show sample entries
        text_col = _get_text_column(historical_df)
        if text_col:
            result += f"=== SAMPLE ENTRIES (First 5) ===\n"
            display_cols = [text_col] + ([level_col] if level_col else [])
            if timestamp_col and timestamp_col in historical_df.columns:
                display_cols.append(timestamp_col)
            # Format text column for RTL display
            display_df = historical_df.head(5)[display_cols].copy()
            if text_col in display_df.columns:
                display_df[text_col] = display_df[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
            result += display_df.to_string()
            result += f"\n\n"
        
        result += f"⚠️ INSTRUCTIONS FOR LLM:\n"
        result += f"Use this historical data to:\n"
        result += f"- Compare with current data to identify trends\n"
        result += f"- Analyze changes over time\n"
        result += f"- Track improvements or declines\n"
        result += f"- Use with analyze_feedback_trends or compare_feedback_periods for detailed analysis\n"
        
        return result
    except Exception as e:
        return f"Error retrieving historical data: {str(e)}"

@tool
def analyze_feedback_trends(start_date: str = None, end_date: str = None, period: str = "month", 
                            compare_periods: bool = False, use_historical: bool = False) -> str:
    """
    Analyze feedback trends over time. Compare feedback patterns, issues, and improvements across different time periods.
    Can use current data (short-term) or historical data (long-term memory) for trend analysis.
    
    Args:
        start_date: Start date for analysis (YYYY-MM-DD format, optional)
        end_date: End date for analysis (YYYY-MM-DD format, optional)
        period: Time period grouping - "day", "week", "month", "quarter", "year" (default: "month")
        compare_periods: If True, compare current period with previous period (default: False)
        use_historical: If True, use historical database (long-term memory). If False, use current CSV data (default: False)
    
    Returns:
        Analysis of feedback trends over time with statistics and insights
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        # Choose data source: historical (long-term) or current (short-term)
        if use_historical:
            df = _get_historical_data(start_date, end_date, limit=10000)
            if len(df) == 0:
                return "Error: No historical data found. Historical database may be empty."
            data_source = "Historical Database (Long-Term Memory)"
        else:
            df = feedback_df.copy()
            data_source = "Current CSV Data (Short-Term Memory)"
        
        timestamp_col = _get_timestamp_column(df)
        if not timestamp_col:
            return f"Error: No timestamp/date column found in the {data_source.lower()}. Cannot perform time-based analysis."
        
        text_col = _get_text_column(df)
        if not text_col:
            return "Error: No text column found in dataframe"
        
        # Find level column
        level_col = None
        for col in df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        # Prepare dataframe with timestamp
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df[df[timestamp_col].notna()].copy()
        
        if len(df) == 0:
            return "Error: No valid timestamps found in the data."
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df[timestamp_col] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df[timestamp_col] <= end_dt]
        
        if len(df) == 0:
            return f"No feedback found in the specified date range."
        
        result = f"=== FEEDBACK TREND ANALYSIS ===\n\n"
        result += f"Data Source: {data_source}\n"
        result += f"Analysis period: {df[timestamp_col].min().strftime('%Y-%m-%d')} to {df[timestamp_col].max().strftime('%Y-%m-%d')}\n"
        result += f"Total entries: {len(df)}\n\n"
        
        # Group by time period
        if period == "day":
            df['period'] = df[timestamp_col].dt.date
        elif period == "week":
            df['period'] = df[timestamp_col].dt.to_period('W')
        elif period == "month":
            df['period'] = df[timestamp_col].dt.to_period('M')
        elif period == "quarter":
            df['period'] = df[timestamp_col].dt.to_period('Q')
        elif period == "year":
            df['period'] = df[timestamp_col].dt.to_period('Y')
        else:
            df['period'] = df[timestamp_col].dt.to_period('M')
        
        # Statistics by period
        agg_dict = {timestamp_col: 'count'}
        if level_col:
            agg_dict[level_col] = ['mean', 'min', 'max']
        period_stats = df.groupby('period').agg(agg_dict).reset_index()
        
        result += f"=== FEEDBACK VOLUME BY {period.upper()} ===\n"
        for _, row in period_stats.iterrows():
            period_str = str(row['period'])
            count = row[(timestamp_col, 'count')]
            if level_col:
                avg_level = row[(level_col, 'mean')]
                result += f"{period_str}: {count} entries (avg level: {avg_level:.2f})\n"
            else:
                result += f"{period_str}: {count} entries\n"
        
        result += f"\n"
        
        # Level distribution trends
        if level_col:
            result += f"=== LEVEL DISTRIBUTION TRENDS ===\n"
            level_trends = df.groupby(['period', level_col]).size().unstack(fill_value=0)
            result += level_trends.to_string()
            result += f"\n\n"
        
        # Compare periods if requested
        if compare_periods:
            periods = sorted(df['period'].unique())
            if len(periods) >= 2:
                current_period = periods[-1]
                previous_period = periods[-2]
                
                current_df = df[df['period'] == current_period]
                previous_df = df[df['period'] == previous_period]
                
                result += f"=== PERIOD COMPARISON ===\n"
                result += f"Previous {period}: {previous_period} ({len(previous_df)} entries)\n"
                result += f"Current {period}: {current_period} ({len(current_df)} entries)\n\n"
                
                if level_col:
                    current_avg = current_df[level_col].mean()
                    previous_avg = previous_df[level_col].mean()
                    change = current_avg - previous_avg
                    result += f"Average Level:\n"
                    result += f"  Previous: {previous_avg:.2f}\n"
                    result += f"  Current: {current_avg:.2f}\n"
                    result += f"  Change: {change:+.2f} ({'↑ Improvement' if change > 0 else '↓ Decline' if change < 0 else '→ Stable'})\n\n"
        
        result += f"\n⚠️ INSTRUCTIONS FOR LLM:\n"
        result += f"Analyze these trends to:\n"
        result += f"- Identify patterns in feedback volume over time\n"
        result += f"- Track changes in level ratings (improvements or declines)\n"
        result += f"- Compare periods to understand what changed\n"
        result += f"- Identify trends in specific issues or complaints\n"
        result += f"- Provide insights on whether feedback is improving or worsening\n"
        result += f"- Note: Use semantic search or theme analysis to understand WHAT issues changed\n"
        
        return result
    except Exception as e:
        return f"Error analyzing trends: {str(e)}"

@tool
def compare_feedback_periods(period1_start: str, period1_end: str, 
                             period2_start: str, period2_end: str, use_historical: bool = True) -> str:
    """
    Compare feedback between two time periods to identify changes, improvements, or new issues.
    Uses historical database (long-term memory) by default for comprehensive trend analysis.
    
    Args:
        period1_start: Start date of first period (YYYY-MM-DD)
        period1_end: End date of first period (YYYY-MM-DD)
        period2_start: Start date of second period (YYYY-MM-DD)
        period2_end: End date of second period (YYYY-MM-DD)
        use_historical: If True, use historical database (long-term memory). If False, use current CSV data (default: True)
    
    Returns:
        Detailed comparison between the two periods
    """
    if feedback_df is None:
        return "Error: Feedback dataframe not initialized. Please ensure data is loaded."
    
    try:
        # Choose data source: historical (long-term) or current (short-term)
        if use_historical:
            # Get historical data covering both periods
            all_dates = [period1_start, period1_end, period2_start, period2_end]
            min_date = min([d for d in all_dates if d])
            max_date = max([d for d in all_dates if d])
            df = _get_historical_data(min_date, max_date, limit=10000)
            if len(df) == 0:
                return "Error: No historical data found. Historical database may be empty."
            data_source = "Historical Database (Long-Term Memory)"
        else:
            df = feedback_df.copy()
            data_source = "Current CSV Data (Short-Term Memory)"
        
        timestamp_col = _get_timestamp_column(df)
        if not timestamp_col:
            return f"Error: No timestamp/date column found in the {data_source.lower()}. Cannot perform time-based comparison."
        
        text_col = _get_text_column(df)
        level_col = None
        for col in df.columns:
            if col.lower() == 'level':
                level_col = col
                break
        
        # Prepare dataframe
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df[df[timestamp_col].notna()].copy()
        
        # Filter periods
        p1_start = pd.to_datetime(period1_start)
        p1_end = pd.to_datetime(period1_end)
        p2_start = pd.to_datetime(period2_start)
        p2_end = pd.to_datetime(period2_end)
        
        period1_df = df[(df[timestamp_col] >= p1_start) & (df[timestamp_col] <= p1_end)]
        period2_df = df[(df[timestamp_col] >= p2_start) & (df[timestamp_col] <= p2_end)]
        
        if len(period1_df) == 0:
            return f"Error: No feedback found in Period 1 ({period1_start} to {period1_end})"
        if len(period2_df) == 0:
            return f"Error: No feedback found in Period 2 ({period2_start} to {period2_end})"
        
        result = f"=== FEEDBACK PERIOD COMPARISON ===\n\n"
        result += f"Data Source: {data_source}\n\n"
        result += f"Period 1: {period1_start} to {period1_end}\n"
        result += f"  Entries: {len(period1_df)}\n"
        if level_col:
            result += f"  Average Level: {period1_df[level_col].mean():.2f}\n"
            result += f"  Level Distribution:\n"
            p1_levels = period1_df[level_col].value_counts().sort_index()
            for level, count in p1_levels.items():
                result += f"    Level {level}: {count} ({count/len(period1_df)*100:.1f}%)\n"
        
        result += f"\nPeriod 2: {period2_start} to {period2_end}\n"
        result += f"  Entries: {len(period2_df)}\n"
        if level_col:
            result += f"  Average Level: {period2_df[level_col].mean():.2f}\n"
            result += f"  Level Distribution:\n"
            p2_levels = period2_df[level_col].value_counts().sort_index()
            for level, count in p2_levels.items():
                result += f"    Level {level}: {count} ({count/len(period2_df)*100:.1f}%)\n"
        
        # Changes
        result += f"\n=== CHANGES ===\n"
        volume_change = len(period2_df) - len(period1_df)
        result += f"Volume: {volume_change:+d} entries ({volume_change/len(period1_df)*100:+.1f}%)\n"
        
        if level_col:
            level_change = period2_df[level_col].mean() - period1_df[level_col].mean()
            result += f"Average Level: {level_change:+.2f} ({'↑ Improvement' if level_change > 0 else '↓ Decline' if level_change < 0 else '→ Stable'})\n"
        
        result += f"\n⚠️ INSTRUCTIONS FOR LLM:\n"
        result += f"Compare these periods to:\n"
        result += f"- Identify what issues improved or worsened\n"
        result += f"- Find new issues that appeared in Period 2\n"
        result += f"- Identify issues that were resolved\n"
        result += f"- Use semantic search or theme analysis to understand WHAT changed\n"
        result += f"- Provide actionable insights on trends\n"
        result += f"\nSample Period 1 feedback (first 10 entries):\n"
        # Format text column for RTL display
        period1_display = period1_df.head(10)[[text_col] + ([level_col] if level_col else [])].copy()
        if text_col in period1_display.columns:
            period1_display[text_col] = period1_display[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
        result += period1_display.to_string()
        result += f"\n\nSample Period 2 feedback (first 10 entries):\n"
        period2_display = period2_df.head(10)[[text_col] + ([level_col] if level_col else [])].copy()
        if text_col in period2_display.columns:
            period2_display[text_col] = period2_display[text_col].apply(lambda x: _format_text_for_display(x) if pd.notna(x) else "")
        result += period2_display.to_string()
        
        return result
    except Exception as e:
        return f"Error comparing periods: {str(e)}"

# List of all tools for easy import
__all__ = [
    'initialize_tools',
    'get_feedback_by_level',
    'get_feedback_statistics',
    'generate_dataframe_report',
    'analyze_feedback_themes',
    'generate_categorized_issues_report',
    'semantic_search_feedback',
    'analyze_feedback_trends',
    'compare_feedback_periods',
    'get_historical_feedback'
]

