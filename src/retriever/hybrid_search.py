import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

# In newer versions of langchain, use langchain_chroma. previous might be langchain_community.vectorstores
import os
from src.config import DB_DIR, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, use_dashscope=False):
        self.db_dir = str(DB_DIR)
        
        # Initialize VectorStore
        # We exclusively use the DashScopeEmbeddings since OpenAIEmbeddings might send token integer arrays which DashScope rejects
        # Make sure DASHSCOPE_API_KEY is configured in the environment (done inside config.py) 
        self.vectorstore = Chroma(
            collection_name="tesla_reports",
            embedding_function=DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME),
            persist_directory=self.db_dir
        )
        
        # BM25 and Document Storage paths
        self.bm25_path = DB_DIR / "bm25.pkl"
        self.docs_path = DB_DIR / "docs.pkl"
        
        self.bm25 = None
        self.documents = []
        
        # Load Sparse index if exists
        self.load_bm25()

    def build_index(self, documents: List[Document]):
        """
        Build and persist both the dense (Chroma) and sparse (BM25) indices.
        """
        MAX_EMBEDDING_LENGTH = 8000  # DashScope text-embedding-v3 limit is 8192, leave safety margin
        
        logger.info(f"Adding {len(documents)} documents to ChromaDB...")
        
        # Validating and sanitizing document content
        valid_documents = []
        truncated_count = 0
        for doc in documents:
            if not isinstance(doc.page_content, str) or not doc.page_content.strip():
                logger.warning(f"Skipping document with invalid or empty page_content.")
                continue
            # Truncate overly long texts to fit within DashScope's embedding input limit
            if len(doc.page_content) > MAX_EMBEDDING_LENGTH:
                doc.page_content = doc.page_content[:MAX_EMBEDDING_LENGTH]
                truncated_count += 1
            valid_documents.append(doc)
                
        logger.info(f"Filtered to {len(valid_documents)} valid documents. ({truncated_count} truncated to {MAX_EMBEDDING_LENGTH} chars)")
        
        # Progress tracking file for batch resumption
        progress_file = DB_DIR / "ingest_progress.txt"
        start_batch = 0
        if progress_file.exists():
            try:
                start_batch = int(progress_file.read_text().strip())
                logger.info(f"Resuming from batch index {start_batch} (previous progress found).")
            except Exception:
                start_batch = 0
        
        # Add documents in batches with progress saving
        batch_size = 25  # Smaller batches to reduce risk of single-batch failure
        for i in range(start_batch, len(valid_documents), batch_size):
            batch = valid_documents[i:i+batch_size]
            try:
                self.vectorstore.add_documents(batch)
                # Save progress after each successful batch
                progress_file.write_text(str(i + batch_size))
                logger.info(f"Added batch {i} to {i+len(batch)}")
            except Exception as e:
                logger.error(f"Failed at batch {i}: {e}")
                raise
        
        # Clean up progress file after full success
        if progress_file.exists():
            progress_file.unlink()
            
        logger.info("Building BM25 Index...")
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.documents = documents
        
        # Persist BM25 and documents
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info("Indices built and persisted successfully.")
        
    def load_bm25(self):
        """Loads BM25 index from disk."""
        if self.bm25_path.exists() and self.docs_path.exists():
            try:
                with open(self.bm25_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded BM25 index with {len(self.documents)} documents.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                
    def get_relevant_documents(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None, top_k: int = 4) -> List[Document]:
        """
        Retrieves top_k documents using both Dense and Sparse retrieval, combined via Reciprocal Rank Fusion.
        """
        # 1. Vector Search
        # langchain Chroma accepts filter={"year": 2022, "report_type": "10-K"}
        dense_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k*2, filter=metadata_filter
        )
        # dense_results is list of (Document, score)
        
        # 2. BM25 Search (Filtered manually since rank_bm25 doesn't support metadata filtering out of the box)
        if self.bm25 is None or not self.documents:
            logger.warning("BM25 index not loaded. Falling back to dense retrieval only.")
            return [doc for doc, score in dense_results][:top_k]
            
        tokenized_query = query.lower().split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # apply metadata filtering for BM25
        sparse_candidates = []
        for i, doc in enumerate(self.documents):
            # check metadata matching
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    # For years we might need complex checks, but for exact match:
                    # also handle $in if needed, but assuming simple equals for now
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
                    
            sparse_candidates.append((doc, doc_scores[i]))
            
        sparse_candidates = sorted(sparse_candidates, key=lambda x: x[1], reverse=True)[:top_k*2]
        
        # 3. Reciprocal Rank Fusion (RRF)
        return self._reciprocal_rank_fusion(dense_results, sparse_candidates, top_k)
        
    def _reciprocal_rank_fusion(self, dense_results, sparse_results, top_k, k=60) -> List[Document]:
        """ Combine results using RRF """
        rrf_score = {}
        
        for rank, (doc, _) in enumerate(dense_results):
            # Using page_content as rough unique identifier for fusion
            doc_id = doc.page_content 
            if doc_id not in rrf_score:
                rrf_score[doc_id] = {'doc': doc, 'score': 0.0}
            rrf_score[doc_id]['score'] += 1.0 / (k + rank + 1)
            
        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = doc.page_content
            if doc_id not in rrf_score:
                rrf_score[doc_id] = {'doc': doc, 'score': 0.0}
            rrf_score[doc_id]['score'] += 1.0 / (k + rank + 1)
            
        # Sort by final RRF score
        fused_results = sorted(list(rrf_score.values()), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in fused_results][:top_k]

if __name__ == "__main__":
    pass
