"""
Codebase Index Module

Provides comprehensive codebase representation for SWE-Bench repos using:
1. Semantic Index - LEANN-style vector search with hub-based pruning
2. Structural Index - Knowledge graph from AST
3. Lexical Index - BM25 full-text search

Usage:
    from agent.codebase_index import CodebaseIndexer, CodebaseRetriever
    
    indexer = CodebaseIndexer(repo_path, index_dir)
    await indexer.build_all_indices()
    
    retriever = CodebaseRetriever(indexer)
    results = await retriever.retrieve("how to authenticate users")
"""

from .code_types import CodeChunk, CodeEntity, CodeRelation, RetrievalResult
from .indexer import CodebaseIndexer
from .retrieval import CodebaseRetriever

__all__ = [
    "CodeChunk",
    "CodeEntity", 
    "CodeRelation",
    "RetrievalResult",
    "CodebaseIndexer",
    "CodebaseRetriever",
]
