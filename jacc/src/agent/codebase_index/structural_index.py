"""
Structural index using knowledge graph.

Represents code structure as a graph where:
- Nodes: Code entities (classes, functions, methods, imports)
- Edges: Relations (calls, imports, inherits, uses, defines)

Enables structural queries like "what calls this function" or
"what classes inherit from X".
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from .code_types import CodeChunk, CodeEntity, CodeRelation, RetrievalResult

logger = logging.getLogger(__name__)


class StructuralIndex:
    """
    Knowledge graph representing code structure.
    
    Provides graph-based code navigation and retrieval through
    entity relationships like calls, inheritance, and imports.
    
    Features:
    - Spreading activation search (memory-graph inspired)
    - Bidirectional traversal
    - PageRank-style importance scoring
    - Entity type filtering
    - Configurable decay factors
    
    Usage:
        index = StructuralIndex()
        index.build_index(entities, relations, chunks)
        
        # Search with spreading activation
        results = index.search(["UserManager", "login"])
        
        # Direct queries
        callers = index.get_callers("save_user")
        subclasses = index.get_subclasses("BaseModel")
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        # Search parameters
        decay_outgoing: float = 0.8,
        decay_incoming: float = 0.6,
        activation_threshold: float = 0.1,
    ):
        """
        Initialize the structural index.
        
        Args:
            storage_path: Path to store index data
            decay_outgoing: Activation decay for outgoing edges (0-1)
            decay_incoming: Activation decay for incoming edges (0-1)
            activation_threshold: Minimum activation to continue traversal
        """
        self.storage_path = storage_path
        self.decay_outgoing = decay_outgoing
        self.decay_incoming = decay_incoming
        self.activation_threshold = activation_threshold
        
        # Graph data
        self.entities: dict[str, CodeEntity] = {}
        self.entity_by_name: dict[str, list[str]] = defaultdict(list)
        self.entity_by_type: dict[str, list[str]] = defaultdict(list)
        
        # Adjacency lists: from_id -> [(to_id, relation_type, weight)]
        self.outgoing: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        self.incoming: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        
        # PageRank-style importance scores
        self.importance: dict[str, float] = {}
        
        # Chunk references
        self.chunks: dict[str, CodeChunk] = {}
        self.entity_to_chunk: dict[str, str] = {}
        
        self._initialized = False
    
    def build_index(
        self,
        entities: list[CodeEntity],
        relations: list[CodeRelation],
        chunks: list[CodeChunk] | None = None,
        compute_importance: bool = True,
    ) -> dict[str, int]:
        """
        Build knowledge graph from parsed entities and relations.
        
        Args:
            entities: List of code entities
            relations: List of relations between entities
            chunks: Optional chunks for result retrieval
            compute_importance: Whether to compute PageRank-style importance
            
        Returns:
            Stats dict with counts
        """
        logger.info(f"Building structural index: {len(entities)} entities, {len(relations)} relations")
        
        # Store entities
        for entity in entities:
            self.entities[entity.id] = entity
            self.entity_by_name[entity.name].append(entity.id)
            self.entity_by_type[entity.entity_type].append(entity.id)
            if entity.chunk_id:
                self.entity_to_chunk[entity.id] = entity.chunk_id
        
        # Store chunks
        if chunks:
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
        
        # Build adjacency lists
        resolved_relations = 0
        unresolved_targets = set()
        
        for relation in relations:
            from_id = self._resolve_entity_id(relation.from_entity_id)
            to_id = self._resolve_entity_id(relation.to_entity_id)
            
            if from_id is None:
                continue
            
            if to_id is None:
                unresolved_targets.add(relation.to_entity_id)
                continue
            
            self.outgoing[from_id].append((to_id, relation.relation_type, relation.weight))
            self.incoming[to_id].append((from_id, relation.relation_type, relation.weight))
            resolved_relations += 1
        
        # Compute importance scores
        if compute_importance:
            self._compute_importance()
        
        self._initialized = True
        
        logger.info(f"Built graph: {len(self.entities)} nodes, {resolved_relations} edges, {len(unresolved_targets)} unresolved")
        
        return {
            "entities": len(self.entities),
            "relations": resolved_relations,
            "unresolved": len(unresolved_targets),
            "entity_types": dict([(t, len(ids)) for t, ids in self.entity_by_type.items()]),
        }
    
    def _resolve_entity_id(self, entity_ref: str) -> str | None:
        """Resolve entity reference to actual ID."""
        # Direct ID match
        if entity_ref in self.entities:
            return entity_ref
        
        # Name lookup
        candidates = self.entity_by_name.get(entity_ref, [])
        if candidates:
            return candidates[0]
        
        # Qualified name lookup (e.g., "module.Class.method")
        parts = entity_ref.split(".")
        if len(parts) > 1:
            # Try just the last part
            short_name = parts[-1]
            candidates = self.entity_by_name.get(short_name, [])
            if candidates:
                return candidates[0]
        
        return None
    
    def _compute_importance(self, iterations: int = 10, damping: float = 0.85):
        """
        Compute PageRank-style importance scores for entities.
        
        Entities that are called/imported by many others get higher scores.
        """
        n = len(self.entities)
        if n == 0:
            return
        
        # Initialize uniform scores
        scores = {eid: 1.0 / n for eid in self.entities}
        
        for _ in range(iterations):
            new_scores = {}
            
            for entity_id in self.entities:
                # Sum incoming relevance
                incoming_sum = 0.0
                for from_id, _, weight in self.incoming.get(entity_id, []):
                    out_degree = len(self.outgoing.get(from_id, []))
                    if out_degree > 0:
                        incoming_sum += scores[from_id] * weight / out_degree
                
                new_scores[entity_id] = (1 - damping) / n + damping * incoming_sum
            
            scores = new_scores
        
        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        self.importance = {eid: s / max_score for eid, s in scores.items()}
    
    def search(
        self,
        query_entities: list[str],
        relation_types: list[str] | None = None,
        entity_types: list[str] | None = None,
        max_depth: int = 2,
        top_k: int = 10,
        use_importance: bool = True,
    ) -> list[RetrievalResult]:
        """
        Search via graph traversal with spreading activation.
        
        Algorithm:
        1. Find entity nodes matching query names
        2. Traverse with decaying activation
        3. Optionally boost by PageRank importance
        4. Collect and rank visited chunks
        
        Args:
            query_entities: Entity names to start from
            relation_types: Filter to specific relation types
            entity_types: Filter to specific entity types
            max_depth: Maximum traversal depth
            top_k: Number of results to return
            use_importance: Boost scores by PageRank importance
            
        Returns:
            List of retrieval results with structural scores
        """
        if not self._initialized:
            return []
        
        # Find starting entities
        start_ids = self._find_entities(query_entities, entity_types)
        
        if not start_ids:
            return []
        
        # BFS with activation decay
        visited: dict[str, float] = {}
        frontier = [(eid, 1.0) for eid in start_ids]
        
        for depth in range(max_depth + 1):
            next_frontier = []
            
            for entity_id, activation in frontier:
                if entity_id in visited:
                    if visited[entity_id] >= activation:
                        continue
                
                visited[entity_id] = activation
                
                # Apply entity type filter
                if entity_types:
                    entity = self.entities.get(entity_id)
                    if entity and entity.entity_type not in entity_types:
                        continue
                
                # Expand outgoing edges
                for neighbor_id, rel_type, weight in self.outgoing.get(entity_id, []):
                    if relation_types and rel_type not in relation_types:
                        continue
                    
                    new_activation = activation * weight * self.decay_outgoing
                    if new_activation > self.activation_threshold:
                        next_frontier.append((neighbor_id, new_activation))
                
                # Expand incoming edges (reverse relations)
                for neighbor_id, rel_type, weight in self.incoming.get(entity_id, []):
                    if relation_types and rel_type not in relation_types:
                        continue
                    
                    new_activation = activation * weight * self.decay_incoming
                    if new_activation > self.activation_threshold:
                        next_frontier.append((neighbor_id, new_activation))
            
            frontier = next_frontier
        
        # Convert to chunk scores
        chunk_scores: dict[str, float] = defaultdict(float)
        
        for entity_id, activation in visited.items():
            chunk_id = self.entity_to_chunk.get(entity_id)
            if chunk_id:
                # Boost by importance if enabled
                importance_boost = self.importance.get(entity_id, 0.5) if use_importance else 1.0
                score = activation * (0.7 + 0.3 * importance_boost)  # Importance adds up to 30%
                chunk_scores[chunk_id] = max(chunk_scores[chunk_id], score)
        
        # Sort and return
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: -x[1])[:top_k]
        
        results = []
        for chunk_id, score in sorted_chunks:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    source="structural",
                    structural_score=score,
                ))
        
        return results
    
    def _find_entities(
        self,
        names: list[str],
        entity_types: list[str] | None = None,
    ) -> set[str]:
        """Find entity IDs matching names, optionally filtered by type."""
        result = set()
        
        for name in names:
            # Exact match
            if name in self.entity_by_name:
                for eid in self.entity_by_name[name]:
                    if entity_types:
                        entity = self.entities.get(eid)
                        if entity and entity.entity_type in entity_types:
                            result.add(eid)
                    else:
                        result.add(eid)
            else:
                # Partial match (case-insensitive)
                name_lower = name.lower()
                for entity_name, entity_ids in self.entity_by_name.items():
                    if name_lower in entity_name.lower():
                        for eid in entity_ids:
                            if entity_types:
                                entity = self.entities.get(eid)
                                if entity and entity.entity_type in entity_types:
                                    result.add(eid)
                            else:
                                result.add(eid)
        
        return result
    
    # =========================================================================
    # Convenience Query Methods
    # =========================================================================
    
    def get_related(
        self,
        entity_name: str,
        relation_type: str | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    ) -> list[tuple[CodeEntity, str, float]]:
        """
        Get entities related to a given entity.
        
        Args:
            entity_name: Name of the source entity
            relation_type: Optional filter for relation type
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of (entity, relation_type, weight) tuples
        """
        results = []
        entity_ids = self.entity_by_name.get(entity_name, [])
        
        for entity_id in entity_ids:
            if direction in ("outgoing", "both"):
                for neighbor_id, rel_type, weight in self.outgoing.get(entity_id, []):
                    if relation_type and rel_type != relation_type:
                        continue
                    neighbor = self.entities.get(neighbor_id)
                    if neighbor:
                        results.append((neighbor, rel_type, weight))
            
            if direction in ("incoming", "both"):
                for neighbor_id, rel_type, weight in self.incoming.get(entity_id, []):
                    if relation_type and rel_type != relation_type:
                        continue
                    neighbor = self.entities.get(neighbor_id)
                    if neighbor:
                        results.append((neighbor, rel_type, weight))
        
        return results
    
    def get_callers(self, function_name: str) -> list[CodeEntity]:
        """Get all entities that call a given function."""
        return [e for e, _, _ in self.get_related(function_name, "calls", "incoming")]
    
    def get_callees(self, function_name: str) -> list[CodeEntity]:
        """Get all entities called by a given function."""
        return [e for e, _, _ in self.get_related(function_name, "calls", "outgoing")]
    
    def get_subclasses(self, class_name: str) -> list[CodeEntity]:
        """Get all classes that inherit from a given class."""
        return [e for e, _, _ in self.get_related(class_name, "inherits", "incoming")]
    
    def get_base_classes(self, class_name: str) -> list[CodeEntity]:
        """Get all base classes of a given class."""
        return [e for e, _, _ in self.get_related(class_name, "inherits", "outgoing")]
    
    def get_imports(self, module_name: str) -> list[CodeEntity]:
        """Get all entities imported by a module."""
        return [e for e, _, _ in self.get_related(module_name, "imports", "outgoing")]
    
    def get_importers(self, entity_name: str) -> list[CodeEntity]:
        """Get all modules that import a given entity."""
        return [e for e, _, _ in self.get_related(entity_name, "imports", "incoming")]
    
    def get_users(self, entity_name: str) -> list[CodeEntity]:
        """Get all entities that use a given entity."""
        return [e for e, _, _ in self.get_related(entity_name, "uses", "incoming")]
    
    def get_definitions(self, entity_name: str) -> list[CodeEntity]:
        """Get all entities defined by a given entity (e.g., methods in a class)."""
        return [e for e, _, _ in self.get_related(entity_name, "defines", "outgoing")]
    
    def get_most_important(
        self,
        entity_type: str | None = None,
        top_k: int = 10,
    ) -> list[tuple[CodeEntity, float]]:
        """
        Get most important entities by PageRank score.
        
        Args:
            entity_type: Optional filter by type
            top_k: Number of results
            
        Returns:
            List of (entity, importance_score) tuples
        """
        candidates = []
        
        for eid, score in self.importance.items():
            entity = self.entities.get(eid)
            if entity:
                if entity_type is None or entity.entity_type == entity_type:
                    candidates.append((entity, score))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(self, path: Path | None = None) -> None:
        """Save index to disk."""
        save_path = path or self.storage_path
        if not save_path:
            return
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            "decay_outgoing": self.decay_outgoing,
            "decay_incoming": self.decay_incoming,
            "activation_threshold": self.activation_threshold,
        }
        (save_path / "config.json").write_text(json.dumps(config))
        
        # Save entities
        entities_data = {eid: e.to_dict() for eid, e in self.entities.items()}
        (save_path / "entities.json").write_text(json.dumps(entities_data))
        
        # Save graph
        graph_data = {
            "outgoing": dict(self.outgoing),
            "incoming": dict(self.incoming),
        }
        (save_path / "graph.json").write_text(json.dumps(graph_data))
        
        # Save importance scores
        (save_path / "importance.json").write_text(json.dumps(self.importance))
        
        # Save chunks
        if self.chunks:
            chunks_data = {cid: c.to_dict() for cid, c in self.chunks.items()}
            (save_path / "chunks.json").write_text(json.dumps(chunks_data))
        
        logger.info(f"Saved structural index to {save_path}")
    
    def load(self, path: Path | None = None) -> bool:
        """Load index from disk."""
        load_path = path or self.storage_path
        if not load_path or not load_path.exists():
            return False
        
        try:
            # Load config
            if (load_path / "config.json").exists():
                config = json.loads((load_path / "config.json").read_text())
                self.decay_outgoing = config.get("decay_outgoing", 0.8)
                self.decay_incoming = config.get("decay_incoming", 0.6)
                self.activation_threshold = config.get("activation_threshold", 0.1)
            
            # Load entities
            entities_data = json.loads((load_path / "entities.json").read_text())
            self.entities = {eid: CodeEntity.from_dict(data) for eid, data in entities_data.items()}
            
            # Rebuild indexes
            for eid, entity in self.entities.items():
                self.entity_by_name[entity.name].append(eid)
                self.entity_by_type[entity.entity_type].append(eid)
                if entity.chunk_id:
                    self.entity_to_chunk[eid] = entity.chunk_id
            
            # Load graph
            graph_data = json.loads((load_path / "graph.json").read_text())
            self.outgoing = defaultdict(list, {
                k: [(n, t, w) for n, t, w in v] 
                for k, v in graph_data.get("outgoing", {}).items()
            })
            self.incoming = defaultdict(list, {
                k: [(n, t, w) for n, t, w in v]
                for k, v in graph_data.get("incoming", {}).items()
            })
            
            # Load importance scores
            if (load_path / "importance.json").exists():
                self.importance = json.loads((load_path / "importance.json").read_text())
            
            # Load chunks
            chunks_path = load_path / "chunks.json"
            if chunks_path.exists():
                chunks_data = json.loads(chunks_path.read_text())
                self.chunks = {cid: CodeChunk.from_dict(data) for cid, data in chunks_data.items()}
            
            self._initialized = True
            logger.info(f"Loaded structural index: {len(self.entities)} entities")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load structural index: {e}")
            return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "initialized": self._initialized,
            "entities": len(self.entities),
            "chunks": len(self.chunks),
            "outgoing_edges": sum(len(v) for v in self.outgoing.values()),
            "incoming_edges": sum(len(v) for v in self.incoming.values()),
            "entity_types": dict([(t, len(ids)) for t, ids in self.entity_by_type.items()]),
            "decay_outgoing": self.decay_outgoing,
            "decay_incoming": self.decay_incoming,
            "activation_threshold": self.activation_threshold,
            "has_importance": len(self.importance) > 0,
        }
