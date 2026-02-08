"""
Main indexer orchestrating all index types.

Provides a unified interface for building and managing:
1. Semantic Index (LEANN-style vector search + BM25 hybrid)
2. Structural Index (Knowledge graph)

Note: Lexical (BM25) search is integrated into SemanticIndex for hybrid search.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .code_types import CodeChunk, CodeEntity, CodeRelation
from .code_parser import CodeParser
from .semantic_index import SemanticIndex, CodeEmbeddings
from .structural_index import StructuralIndex

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """
    Main interface for indexing a codebase.
    
    Creates and manages indices for a repository:
    - SemanticIndex: LEANN-style HNSW + BM25 hybrid search
    - StructuralIndex: Knowledge graph with PageRank
    
    Usage:
        indexer = CodebaseIndexer(repo_path, index_dir)
        await indexer.build_all_indices()
        
        retriever = CodebaseRetriever(indexer)
        results = await retriever.retrieve("query")
    """
    
    def __init__(
        self,
        repo_path: Path,
        index_dir: Path,
        repo_id: str | None = None,
        embeddings: CodeEmbeddings | None = None,
        # Semantic index options
        prune_embeddings: bool = True,
        enable_bm25: bool = True,
        bm25_weight: float = 0.3,
    ):
        """
        Initialize the codebase indexer.
        
        Args:
            repo_path: Path to the repository root
            index_dir: Directory to store index data
            repo_id: Optional repository identifier
            embeddings: Optional embedding model
            prune_embeddings: Use LEANN mode (True) or traditional (False)
            enable_bm25: Enable BM25 hybrid search in semantic index
            bm25_weight: Weight for BM25 in hybrid search
        """
        self.repo_path = Path(repo_path)
        self.index_dir = Path(index_dir)
        self.repo_id = repo_id or self.repo_path.name
        
        # Index options
        self.prune_embeddings = prune_embeddings
        self.enable_bm25 = enable_bm25
        self.bm25_weight = bm25_weight
        
        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Code parser
        self.parser = CodeParser(repo_id=self.repo_id)
        
        # Embeddings
        self._embeddings = embeddings or CodeEmbeddings()
        self._embeddings_initialized = False
        
        # Indices (lazy initialized)
        self._semantic_index: SemanticIndex | None = None
        self._structural_index: StructuralIndex | None = None
        
        # Parsed data
        self._chunks: list[CodeChunk] = []
        self._entities: list[CodeEntity] = []
        self._relations: list[CodeRelation] = []
        
        self._indexed = False
    
    @property
    def semantic_index(self) -> SemanticIndex | None:
        return self._semantic_index
    
    @property
    def structural_index(self) -> StructuralIndex | None:
        return self._structural_index
    
    @property
    def chunks(self) -> list[CodeChunk]:
        return self._chunks
    
    @property
    def is_indexed(self) -> bool:
        return self._indexed
    
    async def build_all_indices(
        self,
        extensions: set[str] | None = None,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """
        Build all indices for the repository.
        
        Args:
            extensions: Optional set of file extensions to include
            max_files: Optional maximum number of files to parse
            
        Returns:
            Stats dict with counts for each index
        """
        logger.info(f"Building indices for {self.repo_path}...")
        
        # Parse codebase
        parse_stats = await self._parse_codebase(extensions, max_files)
        
        if not self._chunks:
            logger.warning("No code chunks found in repository")
            return {"parse": parse_stats, "semantic": {}, "structural": {}}
        
        # Build semantic index (includes BM25)
        semantic_stats = await self._build_semantic_index()
        
        # Build structural index
        structural_stats = self._build_structural_index()
        
        self._indexed = True
        
        # Save all indices
        self.save()
        
        return {
            "parse": parse_stats,
            "semantic": semantic_stats,
            "structural": structural_stats,
        }
    
    async def _parse_codebase(
        self,
        extensions: set[str] | None = None,
        max_files: int | None = None,
    ) -> dict[str, int]:
        """Parse all code files in the repository."""
        logger.info(f"Parsing codebase: {self.repo_path}")
        
        all_chunks = []
        all_entities = []
        all_relations = []
        
        for chunks, entities, relations in self.parser.parse_directory(
            self.repo_path, 
            extensions=extensions,
            max_files=max_files,
        ):
            all_chunks.extend(chunks)
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        self._chunks = all_chunks
        self._entities = all_entities
        self._relations = all_relations
        
        stats = self.parser.stats
        logger.info(f"Parsed: {stats}")
        
        return stats
    
    async def _build_semantic_index(self) -> dict[str, int]:
        """Build the semantic index with BM25 hybrid search."""
        logger.info("Building semantic index...")
        
        self._semantic_index = SemanticIndex(
            embeddings=self._embeddings,
            storage_path=self.index_dir / "semantic",
            prune_embeddings=self.prune_embeddings,
            enable_bm25=self.enable_bm25,
            bm25_weight=self.bm25_weight,
        )
        
        stats = await self._semantic_index.build_index(self._chunks)
        logger.info(f"Semantic index: {stats}")
        
        return stats
    
    def _build_structural_index(self) -> dict[str, int]:
        """Build the structural index."""
        logger.info("Building structural index...")
        
        self._structural_index = StructuralIndex(
            storage_path=self.index_dir / "structural",
        )
        
        stats = self._structural_index.build_index(
            self._entities,
            self._relations,
            self._chunks,
        )
        logger.info(f"Structural index: {stats}")
        
        return stats
    
    def save(self) -> None:
        """Persist all indices to disk."""
        if self._semantic_index:
            self._semantic_index.save()
        if self._structural_index:
            self._structural_index.save()
        
        logger.info(f"Saved all indices to {self.index_dir}")
    
    async def load(self) -> bool:
        """
        Load pre-built indices from disk.
        
        Returns:
            True if all indices loaded successfully
        """
        # Initialize embeddings
        if not self._embeddings_initialized:
            await self._embeddings.initialize()
            self._embeddings_initialized = True
        
        self._semantic_index = SemanticIndex(
            embeddings=self._embeddings,
            storage_path=self.index_dir / "semantic",
            prune_embeddings=self.prune_embeddings,
            enable_bm25=self.enable_bm25,
            bm25_weight=self.bm25_weight,
        )
        self._structural_index = StructuralIndex(
            storage_path=self.index_dir / "structural",
        )
        
        semantic_loaded = self._semantic_index.load()
        structural_loaded = self._structural_index.load()
        
        loaded = semantic_loaded and structural_loaded
        
        if loaded:
            self._indexed = True
            self._chunks = list(self._semantic_index.chunks.values())
            logger.info(f"Loaded indices from {self.index_dir}")
        else:
            if not semantic_loaded:
                logger.warning("Failed to load semantic index")
            if not structural_loaded:
                logger.warning("Failed to load structural index")
        
        return loaded
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all indices."""
        stats = {
            "repo_path": str(self.repo_path),
            "index_dir": str(self.index_dir),
            "indexed": self._indexed,
            "chunks": len(self._chunks),
            "entities": len(self._entities),
            "relations": len(self._relations),
        }
        
        if self._semantic_index:
            stats["semantic"] = self._semantic_index.get_stats()
        
        if self._structural_index:
            stats["structural"] = self._structural_index.get_stats()
        
        return stats
