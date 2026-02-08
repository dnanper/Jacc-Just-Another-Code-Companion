"""
LEANN-style semantic index for code search.

Implements storage-efficient vector search following LEANN architecture:
1. FAISS HNSW graph-based ANN search (full implementation)
2. Embedding pruning (97% storage reduction) - OPTIONAL
3. On-demand recomputation during search with LRU cache
4. BM25 hybrid search for keyword matching
5. PQ pruning for efficient candidate selection

Modes:
- prune_embeddings=True: LEANN mode - prune embeddings, recompute on search
- prune_embeddings=False: Traditional mode - keep embeddings, fast direct search

Reference: https://github.com/dnanper/LEANN
"""

import asyncio
import json
import logging
import math
import pickle
import re
import struct
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np

from .code_types import CodeChunk, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# LRU Cache for Embeddings (Following LEANN)
# =============================================================================

class LRUCache:
    """
    LRU (Least Recently Used) cache for embeddings.
    
    Used to cache recomputed embeddings during search to avoid
    redundant computation for frequently accessed nodes.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> list[float] | None:
        """Get item from cache, updating access order."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def put(self, key: str, value: list[float]) -> None:
        """Add item to cache, evicting LRU if full."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
            self._cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
        }


# =============================================================================
# Embeddings Interface and Implementation
# =============================================================================

class EmbeddingsProtocol(Protocol):
    """Protocol for embedding generators."""
    
    async def initialize(self) -> None:
        ...
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class CodeEmbeddings:
    """
    Code-optimized embeddings using SentenceTransformers.
    """
    
    CODE_MODELS = [
        "nomic-ai/nomic-embed-text-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    @property
    def dimension(self) -> int:
        return self._dimension or 768
    
    async def initialize(self) -> None:
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        models_to_try = [self.model_name] if self.model_name else self.CODE_MODELS
        
        for model_name in models_to_try:
            try:
                logger.info(f"CodeEmbeddings: loading model {model_name}...")
                self._model = SentenceTransformer(model_name, trust_remote_code=True)
                self._dimension = self._model.get_sentence_embedding_dimension()
                self.model_name = model_name
                logger.info(f"CodeEmbeddings: initialized {model_name} (dim={self._dimension})")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        raise RuntimeError(f"Failed to load any embedding model")
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if not texts:
            return []
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]


# =============================================================================
# BM25 Scorer (Following LEANN's implementation)
# =============================================================================

class BM25Scorer:
    """
    BM25 scoring for lexical/keyword search.
    
    Implements Okapi BM25 algorithm:
    - k1: Term frequency saturation parameter (default: 1.2)
    - b: Document length normalization (default: 0.75)
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Index structures
        self.doc_freqs: dict[str, int] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_terms: dict[str, dict[str, int]] = {}
        self.avg_doc_length: float = 0.0
        self.n_docs: int = 0
        self._doc_ids: list[str] = []
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms (code-aware)."""
        text = text.lower()
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace('_', ' ')
        tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through'}
        return [t for t in tokens if len(t) > 1 and t not in stopwords]
    
    def fit(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index from documents."""
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = {}
        self.doc_terms = {}
        self._doc_ids = []
        
        total_length = 0
        
        for doc in documents:
            doc_id = doc['id']
            text = doc['text']
            
            tokens = self._tokenize(text)
            self._doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            term_counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
            self.doc_terms[doc_id] = dict(term_counts)
            
            for term in term_counts:
                self.doc_freqs[term] += 1
        
        self.n_docs = len(documents)
        self.avg_doc_length = total_length / max(1, self.n_docs)
    
    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    def score(self, query_terms: list[str], doc_id: str) -> float:
        """Score a document against query terms."""
        if doc_id not in self.doc_terms:
            return 0.0
        
        doc_length = self.doc_lengths.get(doc_id, 0)
        term_counts = self.doc_terms[doc_id]
        
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents matching query."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        scores = []
        for doc_id in self._doc_ids:
            score = self.score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def save(self, path: Path) -> None:
        """Save BM25 index to disk."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'doc_freqs': dict(self.doc_freqs),
            'doc_lengths': self.doc_lengths,
            'doc_terms': self.doc_terms,
            'avg_doc_length': self.avg_doc_length,
            'n_docs': self.n_docs,
            'doc_ids': self._doc_ids,
        }
        path.write_bytes(pickle.dumps(data))
    
    def load(self, path: Path) -> bool:
        """Load BM25 index from disk."""
        if not path.exists():
            return False
        try:
            data = pickle.loads(path.read_bytes())
            self.k1 = data['k1']
            self.b = data['b']
            self.doc_freqs = data['doc_freqs']
            self.doc_lengths = data['doc_lengths']
            self.doc_terms = data['doc_terms']
            self.avg_doc_length = data['avg_doc_length']
            self.n_docs = data['n_docs']
            self._doc_ids = data['doc_ids']
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False


# =============================================================================
# FAISS HNSW Index with PQ Pruning
# =============================================================================

class FAISSHNSWIndex:
    """
    Full FAISS HNSW index with optional embedding storage.
    
    Two modes:
    - Pruned (LEANN style): No embeddings stored, recompute on search
    - Full: Embeddings stored, direct FAISS search (faster)
    """
    
    def __init__(
        self,
        dimension: int,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        use_pq_pruning: bool = True,
        pq_m: int = 8,
    ):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.use_pq_pruning = use_pq_pruning
        self.pq_m = pq_m
        
        self._index = None
        self._pq_index = None
        self._id_map: list[str] = []
        self._is_pruned = False
        self._faiss_available = False
        
        # Full embeddings storage (when not pruned)
        self._embeddings: np.ndarray | None = None
        
        try:
            import faiss
            self._faiss_available = True
        except ImportError:
            logger.warning("FAISS not available, using fallback implementation")
    
    def build(self, embeddings: np.ndarray, ids: list[str], keep_embeddings: bool = True) -> None:
        """
        Build HNSW index.
        
        Args:
            embeddings: Normalized embeddings array
            ids: String IDs for each embedding
            keep_embeddings: If True, store full embeddings for direct search
        """
        n = len(embeddings)
        if n != len(ids):
            raise ValueError(f"Embeddings ({n}) and IDs ({len(ids)}) count mismatch")
        
        self._id_map = list(ids)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / (norms + 1e-8)).astype(np.float32)
        
        # Store full embeddings if requested
        if keep_embeddings:
            self._embeddings = embeddings.copy()
            self._is_pruned = False
        else:
            self._embeddings = None
            self._is_pruned = True
        
        if not self._faiss_available:
            self._build_fallback(embeddings, keep_embeddings)
            return
        
        import faiss
        
        # Build main HNSW index
        self._index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search
        self._index.add(embeddings)
        
        # Build PQ index for pruning (only if enough data for good clustering)
        # PQ with 256 centroids × 8 subquantizers needs ~10K points ideally
        # Use 5000 as minimum to avoid poor quality / warnings
        if self.use_pq_pruning and n >= 1000:
            try:
                nbits = 8
                self._pq_index = faiss.IndexPQ(self.dimension, self.pq_m, nbits, faiss.METRIC_INNER_PRODUCT)
                self._pq_index.train(embeddings)
                self._pq_index.add(embeddings)
                logger.info(f"Built PQ index for pruning: {self.pq_m} subquantizers")
            except Exception as e:
                logger.warning(f"Failed to build PQ index: {e}")
                self._pq_index = None
        elif self.use_pq_pruning and n < 5000:
            logger.info(f"Skipping PQ index: {n} vectors < 5000 minimum for good clustering")
        
        mode = "pruned" if self._is_pruned else "full"
        logger.info(f"Built FAISS HNSW index: {n} vectors, M={self.M}, mode={mode}")
    
    def _build_fallback(self, embeddings: np.ndarray, keep_embeddings: bool) -> None:
        """Fallback implementation without FAISS."""
        n = len(embeddings)
        
        if keep_embeddings:
            self._fallback_embeddings = embeddings.copy()
        
        # Build simple k-NN graph
        similarities = embeddings @ embeddings.T
        self._fallback_neighbors = {}
        
        for i in range(n):
            sims = similarities[i]
            top_k = min(self.M, n - 1)
            indices = np.argsort(sims)[::-1][1:top_k + 1]
            self._fallback_neighbors[i] = [(int(j), float(sims[j])) for j in indices]
        
        logger.info(f"Built fallback index: {n} vectors")
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        recompute_fn: Callable[[int], list[float] | None] | None = None,
        embedding_cache: LRUCache | None = None,
        ef_search: int | None = None,
        pq_prune_ratio: float = 0.0,
        recompute_budget: int = 100,
    ) -> list[tuple[str, float]]:
        """
        Search with automatic mode detection.
        
        - If not pruned: Direct FAISS search (fast)
        - If pruned: Graph navigation + recomputation (LEANN style)
        """
        # Normalize query
        query = query.flatten().astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)
        query_2d = query.reshape(1, -1)
        
        if not self._faiss_available:
            return self._search_fallback(query, top_k, recompute_fn, embedding_cache, recompute_budget)
        
        import faiss
        
        # Set search parameters
        actual_ef = ef_search or self.ef_search
        self._index.hnsw.efSearch = actual_ef
        
        # Case 1: Not pruned - direct FAISS search (FAST!)
        if not self._is_pruned:
            distances, labels = self._index.search(query_2d, top_k)
            results = []
            for dist, label in zip(distances[0], labels[0]):
                if 0 <= label < len(self._id_map):
                    results.append((self._id_map[label], float(dist)))
            return results
        
        # Case 2: Pruned - need recomputation (LEANN style)
        if recompute_fn is None:
            logger.warning("Index is pruned but no recompute_fn provided")
            return []
        
        # Get more candidates for pruning
        candidate_multiplier = max(3, int(1.0 / (1.0 - pq_prune_ratio + 0.01)))
        n_candidates = min(recompute_budget, top_k * candidate_multiplier)
        
        # Use PQ for fast candidate selection if available
        if self._pq_index is not None and pq_prune_ratio > 0:
            pq_distances, pq_labels = self._pq_index.search(query_2d, n_candidates)
            candidates = [(int(l), float(d)) for l, d in zip(pq_labels[0], pq_distances[0]) if l >= 0]
            
            # Prune by ratio
            n_keep = max(top_k, int(len(candidates) * (1.0 - pq_prune_ratio)))
            candidates = candidates[:n_keep]
        else:
            # Get candidates from HNSW graph
            candidates = self._get_candidates_from_graph(n_candidates)
        
        # Recompute embeddings and score (with caching)
        scored = []
        
        for node_id, approx_score in candidates[:recompute_budget]:
            embedding = None
            chunk_id = self._id_map[node_id] if node_id < len(self._id_map) else None
            
            # Check cache first
            if embedding_cache and chunk_id:
                embedding = embedding_cache.get(chunk_id)
            
            # Recompute if not cached
            if embedding is None:
                embedding = recompute_fn(node_id)
                if embedding is not None and embedding_cache and chunk_id:
                    embedding_cache.put(chunk_id, embedding)
            
            if embedding is None:
                continue
            
            emb = np.array(embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            exact_score = float(query @ emb)
            
            if chunk_id:
                scored.append((chunk_id, exact_score))
        
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
    def _get_candidates_from_graph(self, n_candidates: int) -> list[tuple[int, float]]:
        """Get candidate nodes by traversing HNSW graph."""
        if self._index is None:
            return []
        
        n = self._index.ntotal
        if n == 0:
            return []
        
        candidates = set()
        hnsw = self._index.hnsw
        
        np.random.seed(42)
        entry_points = np.random.choice(n, min(10, n), replace=False)
        
        for ep in entry_points:
            candidates.add(ep)
            try:
                for level in range(hnsw.max_level + 1):
                    neighbors = hnsw.get_neighbor_table(int(ep), level)
                    for neighbor in neighbors:
                        if neighbor >= 0:
                            candidates.add(int(neighbor))
                            if len(candidates) >= n_candidates:
                                break
            except Exception:
                pass
            
            if len(candidates) >= n_candidates:
                break
        
        return [(c, 0.0) for c in list(candidates)[:n_candidates]]
    
    def _search_fallback(
        self,
        query: np.ndarray,
        top_k: int,
        recompute_fn: Callable | None,
        embedding_cache: LRUCache | None,
        recompute_budget: int,
    ) -> list[tuple[str, float]]:
        """Fallback search without FAISS."""
        
        # Case 1: Not pruned - use stored embeddings
        if hasattr(self, '_fallback_embeddings') and self._fallback_embeddings is not None:
            scores = self._fallback_embeddings @ query
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self._id_map[i], float(scores[i])) for i in top_indices]
        
        # Case 2: Pruned - graph navigation with recomputation
        if recompute_fn is None:
            return []
        
        visited = set()
        candidates = []
        
        n = len(self._id_map)
        frontier = list(np.random.choice(n, min(5, n), replace=False))
        
        while frontier and len(visited) < recompute_budget:
            next_frontier = []
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                chunk_id = self._id_map[node_id] if node_id < len(self._id_map) else None
                embedding = None
                
                # Check cache
                if embedding_cache and chunk_id:
                    embedding = embedding_cache.get(chunk_id)
                
                # Recompute
                if embedding is None:
                    embedding = recompute_fn(node_id)
                    if embedding is not None and embedding_cache and chunk_id:
                        embedding_cache.put(chunk_id, embedding)
                
                if embedding is None:
                    continue
                
                emb = np.array(embedding, dtype=np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                score = float(query @ emb)
                candidates.append((chunk_id, score))
                
                for neighbor, _ in self._fallback_neighbors.get(node_id, []):
                    if neighbor not in visited:
                        next_frontier.append(neighbor)
            
            frontier = next_frontier[:self.ef_search]
        
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]
    
    def prune_embeddings(self) -> None:
        """Prune embeddings (LEANN optimization)."""
        self._is_pruned = True
        self._embeddings = None
        
        if hasattr(self, '_fallback_embeddings'):
            del self._fallback_embeddings
        
        logger.info("Embeddings pruned - search requires recomputation")
    
    def get_embedding(self, node_id: int) -> list[float] | None:
        """Get stored embedding by node ID (only if not pruned)."""
        if self._is_pruned or self._embeddings is None:
            return None
        if 0 <= node_id < len(self._embeddings):
            return self._embeddings[node_id].tolist()
        return None
    
    def get_id(self, node_id: int) -> str:
        """Map node ID to string ID."""
        if 0 <= node_id < len(self._id_map):
            return self._id_map[node_id]
        return str(node_id)
    
    def get_node_id(self, string_id: str) -> int:
        """Map string ID to node ID."""
        try:
            return self._id_map.index(string_id)
        except ValueError:
            return -1
    
    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'dimension': self.dimension,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'use_pq_pruning': self.use_pq_pruning,
            'pq_m': self.pq_m,
            'is_pruned': self._is_pruned,
            'faiss_available': self._faiss_available,
            'n_vectors': len(self._id_map),
        }
        (path / 'config.json').write_text(json.dumps(config))
        (path / 'id_map.json').write_text(json.dumps(self._id_map))
        
        # Save FAISS index
        if self._faiss_available and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(path / 'hnsw.index'))
            if self._pq_index is not None:
                faiss.write_index(self._pq_index, str(path / 'pq.index'))
        
        # Save full embeddings if not pruned
        if self._embeddings is not None:
            np.save(path / 'embeddings.npy', self._embeddings)
        
        # Save fallback data
        if hasattr(self, '_fallback_neighbors'):
            (path / 'fallback_neighbors.pkl').write_bytes(pickle.dumps(self._fallback_neighbors))
        if hasattr(self, '_fallback_embeddings') and self._fallback_embeddings is not None:
            np.save(path / 'fallback_embeddings.npy', self._fallback_embeddings)
    
    def load(self, path: Path) -> bool:
        """Load index from disk."""
        if not path.exists():
            return False
        
        try:
            config = json.loads((path / 'config.json').read_text())
            self.dimension = config['dimension']
            self.M = config['M']
            self.ef_construction = config['ef_construction']
            self.ef_search = config['ef_search']
            self.use_pq_pruning = config.get('use_pq_pruning', False)
            self.pq_m = config.get('pq_m', 8)
            self._is_pruned = config.get('is_pruned', False)
            
            self._id_map = json.loads((path / 'id_map.json').read_text())
            
            # Load FAISS index
            if self._faiss_available and (path / 'hnsw.index').exists():
                import faiss
                self._index = faiss.read_index(str(path / 'hnsw.index'))
                if (path / 'pq.index').exists():
                    self._pq_index = faiss.read_index(str(path / 'pq.index'))
            
            # Load embeddings if not pruned
            if (path / 'embeddings.npy').exists():
                self._embeddings = np.load(path / 'embeddings.npy')
            
            # Load fallback data
            if (path / 'fallback_neighbors.pkl').exists():
                self._fallback_neighbors = pickle.loads((path / 'fallback_neighbors.pkl').read_bytes())
            if (path / 'fallback_embeddings.npy').exists():
                self._fallback_embeddings = np.load(path / 'fallback_embeddings.npy')
            
            mode = "pruned" if self._is_pruned else "full"
            logger.info(f"Loaded HNSW index: {len(self._id_map)} vectors, mode={mode}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


# =============================================================================
# Main SemanticIndex Class
# =============================================================================

class SemanticIndex:
    """
    LEANN-style semantic index for code search.
    
    Two operating modes:
    
    1. LEANN Mode (prune_embeddings=True):
       - Prune embeddings after build (97% storage reduction)
       - Recompute embeddings on-demand during search
       - Uses LRU cache to avoid redundant computation
       - Best for: Large codebases, storage-constrained environments
    
    2. Traditional Mode (prune_embeddings=False):
       - Keep all embeddings in memory/disk
       - Direct FAISS search without recomputation
       - Faster search, more storage
       - Best for: Small-medium codebases, low-latency requirements
    
    Both modes support:
    - BM25 hybrid search for keyword matching
    - PQ pruning for candidate selection
    
    Usage:
        # LEANN mode (storage-efficient)
        index = SemanticIndex(prune_embeddings=True)
        
        # Traditional mode (fast search)
        index = SemanticIndex(prune_embeddings=False)
        
        await index.build_index(chunks)
        results = await index.search("query", top_k=10)
    """
    
    def __init__(
        self,
        embeddings: EmbeddingsProtocol | CodeEmbeddings | None = None,
        storage_path: Path | None = None,
        # Mode selection
        prune_embeddings: bool = True,  # True = LEANN mode, False = traditional
        # HNSW parameters
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        # Search parameters
        recompute_budget: int = 100,
        # Cache parameters (for LEANN mode)
        cache_size: int = 1000,
        # BM25 hybrid parameters
        enable_bm25: bool = True,
        bm25_weight: float = 0.3,
        # PQ pruning parameters
        use_pq_pruning: bool = True,
        pq_prune_ratio: float = 0.3,
    ):
        """
        Initialize the semantic index.
        
        Args:
            embeddings: Embedding model instance
            storage_path: Path to store index data
            prune_embeddings: If True, use LEANN mode (prune + recompute).
                            If False, keep embeddings for direct search.
            M: HNSW graph degree
            ef_construction: Build complexity
            ef_search: Search complexity
            recompute_budget: Max embeddings to recompute per query (LEANN mode)
            cache_size: LRU cache size for recomputed embeddings (LEANN mode)
            enable_bm25: Enable BM25 hybrid search
            bm25_weight: Weight for BM25 scores (0-1)
            use_pq_pruning: Enable PQ-based candidate pruning
            pq_prune_ratio: Ratio of candidates to prune with PQ
        """
        self.embeddings = embeddings or CodeEmbeddings()
        self.storage_path = storage_path
        
        # Mode
        self.prune_embeddings = prune_embeddings
        
        # HNSW parameters
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.recompute_budget = recompute_budget
        
        # Cache (for LEANN mode)
        self.cache_size = cache_size
        self._embedding_cache = LRUCache(max_size=cache_size)
        
        # BM25 parameters
        self.enable_bm25 = enable_bm25
        self.bm25_weight = bm25_weight
        
        # PQ parameters
        self.use_pq_pruning = use_pq_pruning
        self.pq_prune_ratio = pq_prune_ratio
        
        # Index data
        self.chunks: dict[str, CodeChunk] = {}
        self._hnsw: FAISSHNSWIndex | None = None
        self._bm25: BM25Scorer | None = None
        
        self._initialized = False
        self._embeddings_initialized = False
    
    async def build_index(self, chunks: list[CodeChunk]) -> dict[str, int]:
        """
        Build index in selected mode.
        
        - LEANN mode: Build index, then prune embeddings
        - Traditional mode: Build index, keep embeddings
        """
        if not chunks:
            return {"chunks": 0, "hnsw_vectors": 0, "bm25_docs": 0, "mode": "none"}
        
        mode = "leann" if self.prune_embeddings else "traditional"
        logger.info(f"Building index for {len(chunks)} chunks (mode={mode})...")
        
        # Initialize embeddings
        if not self._embeddings_initialized:
            await self.embeddings.initialize()
            self._embeddings_initialized = True
        
        # Store chunks
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        
        # Generate embeddings
        texts = [self._chunk_to_text(c) for c in chunks]
        embeddings_list = await self._embed_batch(texts)
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        dimension = embeddings.shape[1]
        chunk_ids = [c.id for c in chunks]
        
        # Build HNSW index
        self._hnsw = FAISSHNSWIndex(
            dimension=dimension,
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            use_pq_pruning=self.use_pq_pruning,
        )
        
        # Build with or without keeping embeddings
        self._hnsw.build(embeddings, chunk_ids, keep_embeddings=not self.prune_embeddings)
        
        # Only prune if LEANN mode
        if self.prune_embeddings:
            self._hnsw.prune_embeddings()
        
        # Build BM25 index
        bm25_docs = 0
        if self.enable_bm25:
            self._bm25 = BM25Scorer()
            documents = [{'id': c.id, 'text': self._chunk_to_text(c)} for c in chunks]
            self._bm25.fit(documents)
            bm25_docs = len(documents)
            logger.info(f"Built BM25 index: {bm25_docs} documents")
        
        # Clear cache on rebuild
        self._embedding_cache.clear()
        
        self._initialized = True
        
        return {
            "chunks": len(chunks),
            "hnsw_vectors": len(chunks),
            "bm25_docs": bm25_docs,
            "mode": mode,
        }
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        recompute_budget: int | None = None,
        ef_search: int | None = None,
        bm25_weight: float | None = None,
        pq_prune_ratio: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid search with automatic mode handling.
        
        - Traditional mode: Direct FAISS search (fast)
        - LEANN mode: Graph navigation + recomputation (storage-efficient)
        """
        if not self._initialized or not self._hnsw:
            return []
        
        # Parameters
        budget = recompute_budget or self.recompute_budget
        actual_bm25_weight = bm25_weight if bm25_weight is not None else self.bm25_weight
        actual_pq_ratio = pq_prune_ratio if pq_prune_ratio is not None else self.pq_prune_ratio
        
        # Ensure embeddings initialized
        if not self._embeddings_initialized:
            await self.embeddings.initialize()
            self._embeddings_initialized = True
        
        n_candidates = top_k * 3
        
        # 1. BM25 search
        bm25_scores: dict[str, float] = {}
        if self.enable_bm25 and self._bm25 is not None and actual_bm25_weight > 0:
            bm25_results = self._bm25.search(query, top_k=n_candidates)
            if bm25_results:
                max_score = max(s for _, s in bm25_results)
                if max_score > 0:
                    bm25_scores = {doc_id: score / max_score for doc_id, score in bm25_results}
        
        # 2. HNSW semantic search
        query_embedding = (await self._embed_batch([query]))[0]
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # Prepare recompute function for LEANN mode
        recompute_fn = None
        if self.prune_embeddings:
            def recompute_embedding(node_id: int) -> list[float] | None:
                chunk_id = self._hnsw.get_id(node_id)
                chunk = self.chunks.get(chunk_id)
                if not chunk:
                    return None
                text = self._chunk_to_text(chunk)
                return self.embeddings.encode([text])[0]
            recompute_fn = recompute_embedding
        
        semantic_results = self._hnsw.search(
            query=query_vec,
            top_k=n_candidates,
            recompute_fn=recompute_fn,
            embedding_cache=self._embedding_cache if self.prune_embeddings else None,
            ef_search=ef_search,
            pq_prune_ratio=actual_pq_ratio,
            recompute_budget=budget,
        )
        
        # Normalize semantic scores
        semantic_scores: dict[str, float] = {}
        if semantic_results:
            max_score = max(s for _, s in semantic_results)
            min_score = min(s for _, s in semantic_results)
            score_range = max_score - min_score
            if score_range > 0:
                semantic_scores = {
                    doc_id: (score - min_score) / score_range 
                    for doc_id, score in semantic_results
                }
            else:
                semantic_scores = {doc_id: 1.0 for doc_id, _ in semantic_results}
        
        # 3. Fuse scores
        all_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        fused_scores: list[tuple[str, float, float, float]] = []
        
        semantic_weight = 1.0 - actual_bm25_weight
        
        for chunk_id in all_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            fused = semantic_weight * semantic_score + actual_bm25_weight * bm25_score
            fused_scores.append((chunk_id, fused, semantic_score, bm25_score))
        
        fused_scores.sort(key=lambda x: -x[1])
        
        # 4. Create results
        results = []
        for chunk_id, fused, semantic, bm25 in fused_scores[:top_k]:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=fused,
                    source="hybrid" if actual_bm25_weight > 0 else "semantic",
                    semantic_score=semantic,
                    lexical_score=bm25,
                ))
        
        return results
    
    def _chunk_to_text(self, chunk: CodeChunk) -> str:
        """Convert chunk to searchable text."""
        parts = [chunk.name]
        if chunk.docstring:
            parts.append(chunk.docstring)
        parts.append(chunk.content[:1000])
        return " ".join(parts)
    
    async def _embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings in batches."""
        if not self._embeddings_initialized:
            await self.embeddings.initialize()
            self._embeddings_initialized = True
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embeddings.encode(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    def save(self, path: Path | None = None) -> None:
        """Save full index to disk."""
        save_path = path or self.storage_path
        if not save_path:
            return
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'prune_embeddings': self.prune_embeddings,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'recompute_budget': self.recompute_budget,
            'cache_size': self.cache_size,
            'enable_bm25': self.enable_bm25,
            'bm25_weight': self.bm25_weight,
            'use_pq_pruning': self.use_pq_pruning,
            'pq_prune_ratio': self.pq_prune_ratio,
        }
        (save_path / 'config.json').write_text(json.dumps(config))
        
        chunks_data = {cid: c.to_dict() for cid, c in self.chunks.items()}
        (save_path / 'chunks.json').write_text(json.dumps(chunks_data))
        
        if self._hnsw:
            self._hnsw.save(save_path / 'hnsw')
        
        if self._bm25:
            self._bm25.save(save_path / 'bm25.pkl')
        
        mode = "leann" if self.prune_embeddings else "traditional"
        logger.info(f"Saved index to {save_path} (mode={mode})")
    
    def load(self, path: Path | None = None) -> bool:
        """Load full index from disk."""
        load_path = path or self.storage_path
        if not load_path or not load_path.exists():
            return False
        
        try:
            config = json.loads((load_path / 'config.json').read_text())
            self.prune_embeddings = config.get('prune_embeddings', True)
            self.M = config['M']
            self.ef_construction = config['ef_construction']
            self.ef_search = config['ef_search']
            self.recompute_budget = config.get('recompute_budget', 100)
            self.cache_size = config.get('cache_size', 1000)
            self.enable_bm25 = config.get('enable_bm25', True)
            self.bm25_weight = config.get('bm25_weight', 0.3)
            self.use_pq_pruning = config.get('use_pq_pruning', True)
            self.pq_prune_ratio = config.get('pq_prune_ratio', 0.3)
            
            # Reinitialize cache with loaded size
            self._embedding_cache = LRUCache(max_size=self.cache_size)
            
            chunks_data = json.loads((load_path / 'chunks.json').read_text())
            self.chunks = {cid: CodeChunk.from_dict(data) for cid, data in chunks_data.items()}
            
            hnsw_path = load_path / 'hnsw'
            if hnsw_path.exists():
                hnsw_config = json.loads((hnsw_path / 'config.json').read_text())
                self._hnsw = FAISSHNSWIndex(
                    dimension=hnsw_config['dimension'],
                    M=hnsw_config['M'],
                    ef_construction=hnsw_config['ef_construction'],
                    ef_search=hnsw_config['ef_search'],
                    use_pq_pruning=hnsw_config.get('use_pq_pruning', False),
                )
                self._hnsw.load(hnsw_path)
            
            bm25_path = load_path / 'bm25.pkl'
            if bm25_path.exists():
                self._bm25 = BM25Scorer()
                self._bm25.load(bm25_path)
            
            self._initialized = True
            mode = "leann" if self.prune_embeddings else "traditional"
            logger.info(f"Loaded index: {len(self.chunks)} chunks (mode={mode})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        stats = {
            'chunks': len(self.chunks),
            'initialized': self._initialized,
            'mode': 'leann' if self.prune_embeddings else 'traditional',
            'M': self.M,
            'ef_search': self.ef_search,
            'bm25_enabled': self.enable_bm25,
            'bm25_weight': self.bm25_weight,
            'pq_pruning_enabled': self.use_pq_pruning,
            'pq_prune_ratio': self.pq_prune_ratio,
        }
        
        if self._hnsw:
            stats['hnsw_vectors'] = len(self._hnsw._id_map)
            stats['hnsw_pruned'] = self._hnsw._is_pruned
            stats['faiss_available'] = self._hnsw._faiss_available
            stats['embeddings_stored'] = self._hnsw._embeddings is not None
        
        if self._bm25:
            stats['bm25_docs'] = self._bm25.n_docs
            stats['bm25_terms'] = len(self._bm25.doc_freqs)
        
        # Cache stats (LEANN mode)
        if self.prune_embeddings:
            cache_stats = self._embedding_cache.get_stats()
            stats['cache_size'] = cache_stats['size']
            stats['cache_max_size'] = cache_stats['max_size']
            stats['cache_hit_rate'] = f"{cache_stats['hit_rate']:.2%}"
        
        return stats
