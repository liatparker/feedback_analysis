"""
Custom LangChain Retriever that uses existing caching and vector DB.
Provides automatic retrieval (no agent decision) with natural language chat support.
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional
import sys
import os

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools as feedback_tools

# Access internal functions for direct vector DB access
# These are private but needed for retriever functionality
from tools import (
    _initialize_vector_db,
    _generate_embeddings,
    _get_text_column,
    _get_rag_cache_key,
    _get_rag_cache_result,
    _store_rag_cache_result,
    _find_similar_query_in_rag_cache,
    CACHE_SIMILARITY_THRESHOLD,
    feedback_df
)
import numpy as np

class CachedFeedbackRetriever(BaseRetriever):
    """
    Custom retriever that automatically retrieves feedback using existing caching.
    No agent decision needed - direct retrieval based on query.
    Supports natural language conversation through ConversationalRetrievalChain.
    
    **Why Custom Retriever?**
    
    The custom retriever is needed for QUERY-LEVEL caching (RAG cache), which is
    different from embedding-level caching:
    
    1. **RAG Cache (Query-Level)**: 
       - Caches: query + params → complete results (documents + metadata)
       - Purpose: Avoid re-running entire retrieval pipeline for same query
       - Example: "common complaints" → cached document list
       - This is UNIQUE to the retriever - not available in standard LangChain retrievers
    
    2. **Semantic Query Cache**:
       - Caches: similar queries → cached results
       - Purpose: Reuse results for semantically similar queries
       - Example: "common complaints" vs "what are the issues" → same results
       - This is also UNIQUE to the retriever
    
    3. **Embedding-Level Caching**:
       - Handled by `_generate_embeddings()` (hash cache + semantic cache)
       - Caches: text → embedding vector
       - Purpose: Avoid regenerating embeddings for same/similar text
       - This is NOT retriever-specific - used everywhere
    
    **LangChain's Built-in Caching vs Custom Implementation:**
    
    LangChain DOES have built-in caching (langchain.cache), but it's designed for:
    - LLM call caching (API responses)
    - General-purpose semantic caching
    - External stores (Redis, SQLite, etc.)
    
    Our custom implementation is specifically designed for:
    - Retriever result caching (query → documents, not just LLM responses)
    - Storing query embeddings for semantic query matching
    - Storing document embeddings for future reuse
    - Integration with our existing embedding cache infrastructure
    - LRU eviction with TTL (tailored to our use case)
    
    **Why Custom Instead of LangChain's Cache?**
    1. **Scope**: LangChain's cache is for LLM calls, not retriever results
    2. **Integration**: Our cache integrates with our embedding cache (hash + semantic)
    3. **Semantic Matching**: We store query embeddings to match similar queries
    4. **Document Embeddings**: We cache document embeddings for reuse
    5. **Infrastructure**: Works seamlessly with our existing caching layers
    
    **Could we use LangChain's cache?**
    - Yes, but we'd need to wrap it anyway to cache retriever results
    - We'd lose the integration with our embedding cache
    - We'd need to implement semantic query matching separately
    - Our custom solution is more tailored to our specific needs
    
    **Trade-offs:**
    - Custom: More tailored, better integration, but more code to maintain
    - LangChain: More general-purpose, but less specific to our use case
    
    **Benefits:**
    - Automatic retrieval (no agent deciding which tool)
    - Query-level caching (RAG cache) for instant repeated queries
    - Semantic query matching (similar queries reuse results)
    - Uses existing embedding cache (via _generate_embeddings())
    - Same vector DB and indexing
    - Supports natural language queries
    - More robust (deterministic retrieval)
    """
    
    def __init__(self, top_k: int = 10, similarity_threshold: float = 0.3):
        super().__init__()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Automatically retrieve relevant feedback documents.
        Follows the exact architecture flow:
        1. RAG Cache (wraps everything) - check first
        2. Generate query embedding (caching built into _generate_embeddings())
        3. Semantic Cache (query embedding vs previous queries) → return cached answer
        4. Vector DB Retriever (query embedding vs docs)
        5. Cache document embeddings (caching built into _generate_embeddings())
        6. RAG Cache (save result + embeddings)
        
        Note: Embedding-level caching (hash cache, semantic cache) is handled
        by _generate_embeddings() - no need to duplicate in retriever!
        The retriever only handles query-level caching (RAG cache).
        
        This is AUTOMATIC - no agent decision needed!
        
        Args:
            query: Natural language query (e.g., "common complaints", "file upload issues")
            run_manager: Callback manager for retriever run
        
        Returns:
            List of Document objects with feedback text and metadata
        """
        if feedback_df is None:
            return [Document(page_content="Feedback data not loaded.", metadata={})]
        
        # ============================================================
        # LAYER 1: RAG Cache (wraps everything) - Check first
        # ============================================================
        rag_cache_key = _get_rag_cache_key(
            query=query,
            top_k=self.top_k,
            filter_by_level=False,
            similarity_threshold=self.similarity_threshold
        )
        cached_result = _get_rag_cache_result(rag_cache_key)
        
        # If RAG cache hit (exact query match), return immediately
        if cached_result is not None:
            return self._parse_results_string(cached_result)
        
        # ============================================================
        # LAYER 2: Generate query embedding (with built-in caching)
        # ============================================================
        # _generate_embeddings() already handles:
        # - Level 1: Hash cache check
        # - Level 2: Semantic cache check
        # - Embedding generation (if cache miss)
        # - Caching the result
        # No need to duplicate caching logic here!
        query_embedding = _generate_embeddings(
            [query],
            use_smart_cache=True,
            use_semantic_cache=True
        )
        
        # ============================================================
        # LAYER 3: Semantic Cache (query embedding vs previous queries)
        # ============================================================
        # This is unique to the retriever - checks if a similar query
        # was asked before (using RAG cache with query embeddings)
        from tools import _find_similar_query_in_rag_cache
        similar_query_result = _find_similar_query_in_rag_cache(
            query_embedding,
            threshold=CACHE_SIMILARITY_THRESHOLD
        )
        
        # If similar query found in RAG cache, return cached answer
        if similar_query_result is not None:
            return self._parse_results_string(similar_query_result)
        
        # ============================================================
        # LAYER 4: Vector DB Retriever (query embedding vs docs)
        # ============================================================
        vector_collection = _initialize_vector_db()
        text_col = _get_text_column(feedback_df)
        
        # Get non-empty entries
        non_empty = feedback_df[feedback_df[text_col].notna() & 
                               (feedback_df[text_col].astype(str).str.strip() != '')].copy()
        
        if len(non_empty) == 0:
            return [Document(page_content="No feedback text found.", metadata={})]
        
        # Query vector database (query embedding vs document embeddings)
        query_results = vector_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(self.top_k * 3, len(non_empty)),
            include=['documents', 'metadatas', 'distances']
        )
        
        if not query_results['ids'] or len(query_results['ids'][0]) == 0:
            return [Document(page_content="No relevant feedback found.", metadata={})]
        
        # Process results
        result_ids = query_results['ids'][0]
        result_distances = query_results['distances'][0]
        result_documents = query_results['documents'][0]
        result_metadatas = query_results['metadatas'][0]
        
        # Convert distances to similarities and filter
        documents = []
        document_texts_for_caching = []  # Store document texts for embedding caching
        
        for result_id, distance, doc, metadata in zip(result_ids, result_distances, 
                                                      result_documents, result_metadatas):
            similarity = 1 - distance
            if similarity < self.similarity_threshold:
                continue
            
            # Get dataframe index
            df_idx = int(result_id.split('_')[1])
            
            # Create Document
            doc_metadata = {
                'similarity': similarity,
                'df_index': df_idx
            }
            if 'level' in metadata:
                doc_metadata['level'] = metadata['level']
            
            documents.append(
                Document(
                    page_content=doc,
                    metadata=doc_metadata
                )
            )
            document_texts_for_caching.append(doc)  # Store for caching
        
        # Sort by similarity and take top_k
        documents.sort(key=lambda x: x.metadata.get('similarity', 0), reverse=True)
        documents = documents[:self.top_k]
        document_texts_for_caching = [doc.page_content for doc in documents]  # Update to match filtered docs
        
        # ============================================================
        # LAYER 5: Cache Document Embeddings (both sides caching)
        # ============================================================
        # Generate/cache embeddings for retrieved documents
        # _generate_embeddings() handles all caching automatically:
        # - Checks hash cache (Level 1)
        # - Checks semantic cache (Level 2)
        # - Generates if needed
        # - Caches the result
        document_embeddings = []
        if document_texts_for_caching:
            for doc_text in document_texts_for_caching:
                # Caching is built into _generate_embeddings() - no need to duplicate!
                doc_embedding = _generate_embeddings(
                    [doc_text],
                    use_smart_cache=True,
                    use_semantic_cache=True
                )
                document_embeddings.append(doc_embedding)
        
        # ============================================================
        # LAYER 6: RAG Cache (save result + query embedding + document embeddings)
        # ============================================================
        # Note: LLM Generation happens in ConversationalRetrievalChain, not here
        # This retriever returns documents, chain handles LLM generation
        
        # Store in RAG cache for next time (with both query and document embeddings)
        if documents:
            # Create result string for caching
            result_str = f"Found {len(documents)} results for query: {query}\n"
            for idx, doc in enumerate(documents, 1):
                result_str += f"Result {idx} (Similarity: {doc.metadata.get('similarity', 0):.3f}):\n"
                if 'level' in doc.metadata:
                    result_str += f"  Level: {doc.metadata['level']}\n"
                result_str += f"  Text: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}\n\n"
            
            # Store result + query embedding + document embeddings in RAG cache
            from tools import _store_rag_cache_result
            _store_rag_cache_result(
                rag_cache_key, 
                result_str, 
                query_embedding=query_embedding,
                document_embeddings=document_embeddings if document_embeddings else None
            )
        
        return documents if documents else [
            Document(page_content="No relevant feedback found.", metadata={})
        ]
    
    def _parse_results_string(self, result_str: str) -> List[Document]:
        """Parse result string from RAG cache into Document objects"""
        documents = []
        lines = result_str.split('\n')
        current_doc = None
        current_text = []
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('Result') and 'Similarity:' in line_stripped:
                if current_doc is not None and current_text:
                    current_doc['text'] = ' '.join(current_text).strip()
                    if current_doc['text']:
                        documents.append(current_doc)
                
                try:
                    similarity_part = line_stripped.split('Similarity:')[1].split(')')[0].strip()
                    similarity = float(similarity_part)
                except:
                    similarity = 0.0
                
                current_doc = {'text': '', 'metadata': {'similarity': similarity}}
                current_text = []
            
            elif line_stripped.startswith('Level:') and current_doc is not None:
                level = line_stripped.replace('Level:', '').strip()
                current_doc['metadata']['level'] = level
            
            elif line_stripped.startswith('Text:') and current_doc is not None:
                text_content = line_stripped.replace('Text:', '').strip()
                if text_content:
                    current_text = [text_content]
            
            elif current_text and line_stripped and not line_stripped.startswith('Result'):
                current_text.append(line_stripped)
        
        if current_doc is not None and current_text:
            current_doc['text'] = ' '.join(current_text).strip()
            if current_doc['text']:
                documents.append(current_doc)
        
        return [
            Document(page_content=doc['text'], metadata=doc['metadata'])
            for doc in documents
        ] if documents else [Document(page_content="No relevant feedback found.", metadata={})]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Async version - same as sync for now"""
        return self._get_relevant_documents(query, run_manager=run_manager)

