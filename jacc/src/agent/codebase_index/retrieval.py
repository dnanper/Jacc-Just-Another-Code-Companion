"""
Unified retrieval across all codebase indices.

Provides fusion ranking using Reciprocal Rank Fusion (RRF) to combine
results from semantic (+ BM25 hybrid) and structural retrieval.
"""

import asyncio
import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .code_types import CodeChunk, RetrievalResult

if TYPE_CHECKING:
    from .indexer import CodebaseIndexer

logger = logging.getLogger(__name__)

# Default fusion weights for RRF
DEFAULT_WEIGHTS = {
    "semantic": 0.6,   # Includes BM25 hybrid internally
    "structural": 0.4,
}

# RRF constant (standard value from literature)
RRF_K = 60


class CodebaseRetriever:
    """
    Unified retrieval across all indices with fusion ranking.
    
    Combines results from:
    - Semantic search (LEANN + BM25 hybrid)
    - Structural search (Knowledge graph)
    
    Uses Reciprocal Rank Fusion (RRF) for robust ranking.
    
    Usage:
        retriever = CodebaseRetriever(indexer)
        
        # Fused retrieval
        results = await retriever.retrieve("how to authenticate users")
        
        # With entity hints for structural search
        results = await retriever.retrieve("login flow", entity_hints=["login", "authenticate"])
    """
    
    def __init__(
        self,
        indexer: "CodebaseIndexer",
        weights: dict[str, float] | None = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            indexer: CodebaseIndexer with built indices
            weights: Optional custom fusion weights (semantic, structural)
        """
        self.indexer = indexer
        self.weights = weights or DEFAULT_WEIGHTS.copy()
    
    async def retrieve(
        self,
        query: str,
        entity_hints: list[str] | None = None,
        top_k: int = 10,
        methods: list[str] | None = None,
        bm25_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Parallel retrieval from all indices with RRF fusion.
        
        Args:
            query: Natural language or code query
            entity_hints: Optional entity names for structural search
            top_k: Number of results to return
            methods: Optional list of methods to use ("semantic", "structural")
                    Default: all methods
            bm25_weight: Optional override for BM25 weight in semantic search
            
        Returns:
            List of fused retrieval results
        """
        if not self.indexer.is_indexed:
            logger.warning("Indexer not built, attempting to load...")
            if not await self.indexer.load():
                logger.error("Failed to load indices")
                return []
        
        methods = methods or list(self.weights.keys())
        
        # Run retrieval methods in parallel
        tasks = []
        method_names = []
        
        # Increase per-method limit for better fusion
        per_method_k = max(top_k * 2, 20)
        
        if "semantic" in methods and self.indexer.semantic_index:
            tasks.append(self._retrieve_semantic(query, per_method_k, bm25_weight))
            method_names.append("semantic")
        
        if "structural" in methods and self.indexer.structural_index:
            # Use entity hints or extract from query
            entities = entity_hints or self._extract_entities_from_query(query)
            if entities:
                tasks.append(self._retrieve_structural(entities, per_method_k))
                method_names.append("structural")
        
        if not tasks:
            return []
        
        # Execute in parallel
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results by method
        method_results: dict[str, list[RetrievalResult]] = {}
        for method_name, results in zip(method_names, results_list):
            if isinstance(results, Exception):
                logger.error(f"{method_name} retrieval failed: {results}")
                continue
            method_results[method_name] = results
        
        # If only one method returned results, return directly
        if len(method_results) == 1:
            method_name = list(method_results.keys())[0]
            results = method_results[method_name][:top_k]
            for r in results:
                r.source = "fused"
                r.metadata["methods"] = [method_name]
            return results
        
        # Fuse results using RRF
        return self._fuse_results(method_results, top_k)
    
    async def _retrieve_semantic(
        self, 
        query: str, 
        top_k: int,
        bm25_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """Run semantic retrieval (includes BM25 hybrid)."""
        try:
            kwargs = {"top_k": top_k}
            if bm25_weight is not None:
                kwargs["bm25_weight"] = bm25_weight
            
            return await self.indexer.semantic_index.search(query, **kwargs)
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    async def _retrieve_structural(
        self,
        entities: list[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Run structural retrieval."""
        try:
            return self.indexer.structural_index.search(entities, top_k=top_k)
        except Exception as e:
            logger.error(f"Structural search error: {e}")
            return []
    
    def _extract_entities_from_query(self, query: str) -> list[str]:
        """
        Extract potential entity names from query.
        
        Uses heuristics to find code-like identifiers:
        - PascalCase (class names)
        - snake_case (function names)
        - Quoted identifiers
        """
        entities = []
        
        # PascalCase (e.g., UserManager, HttpClient)
        pascal_pattern = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b')
        entities.extend(pascal_pattern.findall(query))
        
        # snake_case (e.g., get_user, authenticate_request)
        snake_pattern = re.compile(r'\b[a-z]+(?:_[a-z]+)+\b')
        entities.extend(snake_pattern.findall(query))
        
        # Quoted identifiers (e.g., `login`, 'authenticate')
        quoted_pattern = re.compile(r'[`\'\"]([\w_]+)[`\'\"]')
        entities.extend(quoted_pattern.findall(query))
        
        # camelCase (e.g., getUserById)
        camel_pattern = re.compile(r'\b[a-z]+(?:[A-Z][a-z]+)+\b')
        entities.extend(camel_pattern.findall(query))
        
        return list(set(entities))
    
    def _fuse_results(
        self,
        method_results: dict[str, list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).
        
        RRF Score = sum(weight_i / (k + rank_i)) for each method i
        
        This is robust to different score scales across methods.
        """
        # Calculate RRF scores
        chunk_scores: dict[str, float] = defaultdict(float)
        chunk_results: dict[str, RetrievalResult] = {}
        
        for method, results in method_results.items():
            weight = self.weights.get(method, 1.0)
            
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.id
                
                # RRF contribution
                rrf_score = weight / (RRF_K + rank)
                chunk_scores[chunk_id] += rrf_score
                
                # Keep best result per chunk (with all scores)
                if chunk_id not in chunk_results:
                    chunk_results[chunk_id] = RetrievalResult(
                        chunk=result.chunk,
                        score=0.0,  # Will be set to RRF score
                        source="fused",
                        metadata={"methods": [], "ranks": {}},
                    )
                
                # Store per-method scores and ranks
                if method == "semantic":
                    chunk_results[chunk_id].semantic_score = result.score
                elif method == "structural":
                    chunk_results[chunk_id].structural_score = result.score
                
                chunk_results[chunk_id].metadata["methods"].append(method)
                chunk_results[chunk_id].metadata["ranks"][method] = rank
        
        # Sort by RRF score
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: -chunk_scores[x])
        
        # Build final results
        final_results = []
        for chunk_id in sorted_ids[:top_k]:
            result = chunk_results[chunk_id]
            result.score = chunk_scores[chunk_id]
            result.metadata["methods"] = list(set(result.metadata["methods"]))
            final_results.append(result)
        
        return final_results
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    async def retrieve_semantic_only(
        self, 
        query: str, 
        top_k: int = 10,
        bm25_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using only semantic search (includes BM25 hybrid)."""
        return await self.retrieve(query, top_k=top_k, methods=["semantic"], bm25_weight=bm25_weight)
    
    async def retrieve_semantic_pure(
        self, 
        query: str, 
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve using pure semantic search (no BM25)."""
        return await self.retrieve(query, top_k=top_k, methods=["semantic"], bm25_weight=0.0)
    
    async def retrieve_structural_only(
        self,
        entity_hints: list[str],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve using only structural search."""
        if not self.indexer.structural_index:
            return []
        return self.indexer.structural_index.search(entity_hints, top_k=top_k)
    
    def get_chunk_by_id(self, chunk_id: str) -> CodeChunk | None:
        """Get a specific chunk by ID."""
        if self.indexer.semantic_index:
            return self.indexer.semantic_index.chunks.get(chunk_id)
        return None
    
    def get_file_chunks(self, file_path: str) -> list[CodeChunk]:
        """Get all chunks from a specific file."""
        if not self.indexer.semantic_index:
            return []
        
        return [
            chunk for chunk in self.indexer.semantic_index.chunks.values()
            if chunk.file_path == file_path or chunk.file_path.endswith(file_path)
        ]
    
    def get_callers(self, function_name: str) -> list[Any]:
        """Get callers of a function via structural index."""
        if not self.indexer.structural_index:
            return []
        return self.indexer.structural_index.get_callers(function_name)
    
    def get_callees(self, function_name: str) -> list[Any]:
        """Get callees of a function via structural index."""
        if not self.indexer.structural_index:
            return []
        return self.indexer.structural_index.get_callees(function_name)
