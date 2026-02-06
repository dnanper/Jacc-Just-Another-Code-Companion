"""
Data types for codebase representation.

Defines core data structures for code chunks, entities, relations,
and retrieval results used across all indexing methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Any
from uuid import UUID, uuid4


@dataclass
class CodeChunk:
    """
    A semantic unit of code (function, class, method, etc.).
    
    This is the primary unit for semantic and lexical indexing.
    Each chunk represents a meaningful, searchable piece of code.
    """
    id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: Literal["function", "class", "method", "module", "block"]
    name: str
    docstring: str | None = None
    embedding: list[float] | None = None
    is_hub: bool = False  # LEANN hub node (stores embedding)
    repo_id: str | None = None
    
    @classmethod
    def create(
        cls,
        file_path: str,
        start_line: int,
        end_line: int,
        content: str,
        chunk_type: Literal["function", "class", "method", "module", "block"],
        name: str,
        docstring: str | None = None,
        repo_id: str | None = None,
    ) -> "CodeChunk":
        """Factory method to create a new CodeChunk with generated ID."""
        return cls(
            id=str(uuid4()),
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            chunk_type=chunk_type,
            name=name,
            docstring=docstring,
            repo_id=repo_id,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "docstring": self.docstring,
            "is_hub": self.is_hub,
            "repo_id": self.repo_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            content=data["content"],
            chunk_type=data["chunk_type"],
            name=data["name"],
            docstring=data.get("docstring"),
            is_hub=data.get("is_hub", False),
            repo_id=data.get("repo_id"),
        )


@dataclass
class CodeEntity:
    """
    Entity in knowledge graph (class, function, variable, import).
    
    Entities are nodes in the structural index knowledge graph.
    They represent named code elements that can be referenced.
    """
    id: str
    name: str
    entity_type: Literal["class", "function", "method", "import", "variable", "constant"]
    file_path: str
    line_number: int
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None  # Reference to containing CodeChunk
    repo_id: str | None = None
    
    @classmethod
    def create(
        cls,
        name: str,
        entity_type: Literal["class", "function", "method", "import", "variable", "constant"],
        file_path: str,
        line_number: int,
        metadata: dict[str, Any] | None = None,
        chunk_id: str | None = None,
        repo_id: str | None = None,
    ) -> "CodeEntity":
        """Factory method to create a new CodeEntity with generated ID."""
        return cls(
            id=str(uuid4()),
            name=name,
            entity_type=entity_type,
            file_path=file_path,
            line_number=line_number,
            metadata=metadata or {},
            chunk_id=chunk_id,
            repo_id=repo_id,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "repo_id": self.repo_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeEntity":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data["entity_type"],
            file_path=data["file_path"],
            line_number=data["line_number"],
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id"),
            repo_id=data.get("repo_id"),
        )


@dataclass
class CodeRelation:
    """
    Relationship between entities in knowledge graph.
    
    Represents edges in the structural index, capturing code dependencies
    like function calls, imports, inheritance, etc.
    """
    from_entity_id: str
    to_entity_id: str
    relation_type: Literal["calls", "imports", "inherits", "uses", "defines", "overrides"]
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "from_entity_id": self.from_entity_id,
            "to_entity_id": self.to_entity_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeRelation":
        """Create from dictionary."""
        return cls(
            from_entity_id=data["from_entity_id"],
            to_entity_id=data["to_entity_id"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """
    Unified result from any retrieval method.
    
    Provides a common interface for results from semantic, structural,
    or lexical retrieval, with source attribution and scoring.
    """
    chunk: CodeChunk
    score: float
    source: Literal["semantic", "structural", "lexical", "fused"]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Additional context from different retrieval methods
    semantic_score: float | None = None
    structural_score: float | None = None  
    lexical_score: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "semantic_score": self.semantic_score,
            "structural_score": self.structural_score,
            "lexical_score": self.lexical_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """Create from dictionary."""
        return cls(
            chunk=CodeChunk.from_dict(data["chunk"]),
            score=data["score"],
            source=data["source"],
            metadata=data.get("metadata", {}),
            semantic_score=data.get("semantic_score"),
            structural_score=data.get("structural_score"),
            lexical_score=data.get("lexical_score"),
        )


@dataclass
class ChunkSimilarity:
    """
    Similarity edge in LEANN graph.
    
    Represents pre-computed similarity between two code chunks,
    used for graph-based navigation in semantic search.
    """
    chunk_id_1: str
    chunk_id_2: str
    similarity: float
    
    def __post_init__(self):
        # Ensure canonical ordering (chunk_id_1 < chunk_id_2)
        if self.chunk_id_1 > self.chunk_id_2:
            self.chunk_id_1, self.chunk_id_2 = self.chunk_id_2, self.chunk_id_1


@dataclass
class RepoExperience:
    """
    A learned experience from working with a repo.
    
    Stores patterns, pitfalls, conventions, and solutions
    discovered while working on a specific repository.
    """
    id: str
    repo_name: str  # e.g., "django/django", "scikit-learn/scikit-learn"
    experience_type: Literal["pattern", "pitfall", "convention", "solution"]
    content: str
    context: str | None = None  # What triggered this learning
    embedding: list[float] | None = None
    confidence: float = 0.5
    use_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        repo_name: str,
        experience_type: Literal["pattern", "pitfall", "convention", "solution"],
        content: str,
        context: str | None = None,
    ) -> "RepoExperience":
        """Factory method to create a new RepoExperience."""
        now = datetime.utcnow()
        return cls(
            id=str(uuid4()),
            repo_name=repo_name,
            experience_type=experience_type,
            content=content,
            context=context,
            created_at=now,
            updated_at=now,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "repo_name": self.repo_name,
            "experience_type": self.experience_type,
            "content": self.content,
            "context": self.context,
            "confidence": self.confidence,
            "use_count": self.use_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ExperienceQuery:
    """Query for retrieving relevant experiences."""
    repo_name: str
    problem_context: str
    file_types: list[str] | None = None
    experience_types: list[str] | None = None
