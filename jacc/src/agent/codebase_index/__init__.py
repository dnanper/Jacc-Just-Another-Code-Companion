"""
Codebase Index Module

Provides comprehensive codebase representation for SWE-Bench repos using:
1. Semantic Index - LEANN-style HNSW graph search with embedding recomputation
2. Structural Index - Knowledge graph from AST with PageRank importance
3. Lexical Index - BM25 full-text search

Two operating modes for SemanticIndex:
- LEANN Mode (prune_embeddings=True): 97% storage reduction, recompute on search
- Traditional Mode (prune_embeddings=False): Keep embeddings, fast direct search

Features:
- FAISS HNSW for graph-based ANN search
- BM25 hybrid search for keyword matching
- PQ pruning for efficient candidate selection
- LRU cache for recomputed embeddings
- PageRank-style importance scoring for structural index

Reference: https://github.com/dnanper/LEANN

Usage:
    from agent.codebase_index import SemanticIndex, StructuralIndex
    
    # Semantic search
    semantic = SemanticIndex(prune_embeddings=True)
    await semantic.build_index(chunks)
    results = await semantic.search("query", top_k=10)
    
    # Structural search
    structural = StructuralIndex()
    structural.build_index(entities, relations, chunks)
    callers = structural.get_callers("function_name")
"""

# Core types (always available)
from .code_types import CodeChunk, CodeEntity, CodeRelation, RetrievalResult

# Semantic index components
from .semantic_index import SemanticIndex, CodeEmbeddings, BM25Scorer, FAISSHNSWIndex, LRUCache

# Structural index
from .structural_index import StructuralIndex

# Optional components (may not exist yet)
try:
    from .indexer import CodebaseIndexer
except ImportError:
    CodebaseIndexer = None  # type: ignore

try:
    from .retrieval import CodebaseRetriever
except ImportError:
    CodebaseRetriever = None  # type: ignore

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

