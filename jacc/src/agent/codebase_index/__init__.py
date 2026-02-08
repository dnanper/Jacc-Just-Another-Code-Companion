"""
Codebase Index Module

Provides comprehensive codebase representation for SWE-Bench repos using:
1. Semantic Index - LEANN-style HNSW search + BM25 hybrid
2. Structural Index - Knowledge graph from AST with PageRank importance

Two operating modes for SemanticIndex:
- LEANN Mode (prune_embeddings=True): 97% storage reduction, recompute on search
- Traditional Mode (prune_embeddings=False): Keep embeddings, fast direct search

Features:
- FAISS HNSW for graph-based ANN search
- BM25 hybrid search (integrated into SemanticIndex)
- PQ pruning for efficient candidate selection
- LRU cache for recomputed embeddings
- PageRank-style importance scoring for structural index
- RRF fusion for multi-index retrieval

Reference: https://github.com/dnanper/LEANN

Usage:
    from agent.codebase_index import CodebaseIndexer, CodebaseRetriever
    
    # Build indices
    indexer = CodebaseIndexer(repo_path, index_dir)
    await indexer.build_all_indices()
    
    # Retrieve with RRF fusion
    retriever = CodebaseRetriever(indexer)
    results = await retriever.retrieve("how to authenticate users")
    
    # Or use indices directly
    semantic_results = await indexer.semantic_index.search("query")
    structural_results = indexer.structural_index.search(["UserManager"])
"""

# Core types (always available)
from .code_types import CodeChunk, CodeEntity, CodeRelation, RetrievalResult

# Semantic index components
from .semantic_index import SemanticIndex, CodeEmbeddings, BM25Scorer, FAISSHNSWIndex, LRUCache

# Structural index
from .structural_index import StructuralIndex

# Indexer and retriever
from .indexer import CodebaseIndexer
from .retrieval import CodebaseRetriever


__all__ = [
    # Types
    "CodeChunk",
    "CodeEntity", 
    "CodeRelation",
    "RetrievalResult",
    # Semantic index components
    "SemanticIndex",
    "CodeEmbeddings",
    "BM25Scorer",
    "FAISSHNSWIndex",
    "LRUCache",
    # Structural index
    "StructuralIndex",
    # Optional (may be None if not implemented)
    "CodebaseIndexer",
    "CodebaseRetriever",
]

