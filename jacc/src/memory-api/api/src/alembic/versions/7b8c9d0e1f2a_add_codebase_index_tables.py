"""add_codebase_index_tables

Revision ID: 7b8c9d0e1f2a
Revises: 5a366d414dce
Create Date: 2026-02-06 01:00:00.000000

Adds tables for codebase indexing:
- code_chunks: Code chunks for semantic/lexical search
- code_entities: Knowledge graph nodes
- code_relations: Knowledge graph edges
- chunk_similarities: LEANN similarity graph
- repo_experiences: Per-repo learning memory
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "7b8c9d0e1f2a"
down_revision: str | Sequence[str] | None = "5a366d414dce"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create codebase index tables."""

    # ==== CODE_CHUNKS TABLE ====
    op.create_table(
        "code_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("repo_id", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_type", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("docstring", sa.Text(), nullable=True),
        sa.Column("is_hub", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_code_chunks")),
        sa.CheckConstraint(
            "chunk_type IN ('function', 'class', 'method', 'module', 'block')",
            name="code_chunks_type_check"
        ),
    )

    # Add search_vector column for full-text search
    op.execute("""
        ALTER TABLE code_chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, ''))) STORED
    """)

    # Indexes for code_chunks
    op.create_index("idx_code_chunks_repo", "code_chunks", ["repo_id"])
    op.create_index("idx_code_chunks_file", "code_chunks", ["file_path"])
    op.create_index("idx_code_chunks_repo_file", "code_chunks", ["repo_id", "file_path"])
    
    # Vector index for hub embeddings only
    op.execute("""
        CREATE INDEX idx_code_chunks_hub_embedding ON code_chunks 
        USING hnsw(embedding vector_cosine_ops) WHERE is_hub = TRUE
    """)
    
    # Full-text search index
    op.execute("CREATE INDEX idx_code_chunks_search ON code_chunks USING gin(search_vector)")

    # ==== CODE_ENTITIES TABLE ====
    op.create_table(
        "code_entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("repo_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["code_chunks.id"], name="fk_code_entities_chunk", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_code_entities")),
        sa.CheckConstraint(
            "entity_type IN ('class', 'function', 'method', 'import', 'variable', 'constant')",
            name="code_entities_type_check"
        ),
    )

    op.create_index("idx_code_entities_repo", "code_entities", ["repo_id"])
    op.create_index("idx_code_entities_name", "code_entities", ["repo_id", "name"])
    op.create_index("idx_code_entities_type", "code_entities", ["entity_type"])
    op.create_index("idx_code_entities_chunk", "code_entities", ["chunk_id"])

    # ==== CODE_RELATIONS TABLE ====
    op.create_table(
        "code_relations",
        sa.Column("from_entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("to_entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["from_entity_id"], ["code_entities.id"], name="fk_code_relations_from", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_entity_id"], ["code_entities.id"], name="fk_code_relations_to", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("from_entity_id", "to_entity_id", "relation_type", name=op.f("pk_code_relations")),
        sa.CheckConstraint(
            "relation_type IN ('calls', 'imports', 'inherits', 'uses', 'defines', 'overrides')",
            name="code_relations_type_check"
        ),
    )

    op.create_index("idx_code_relations_from", "code_relations", ["from_entity_id"])
    op.create_index("idx_code_relations_to", "code_relations", ["to_entity_id"])
    op.create_index("idx_code_relations_type", "code_relations", ["relation_type"])

    # ==== CHUNK_SIMILARITIES TABLE (LEANN Graph) ====
    op.create_table(
        "chunk_similarities",
        sa.Column("chunk_id_1", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id_2", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id_1"], ["code_chunks.id"], name="fk_similarities_chunk1", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id_2"], ["code_chunks.id"], name="fk_similarities_chunk2", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id_1", "chunk_id_2", name=op.f("pk_chunk_similarities")),
        sa.CheckConstraint("chunk_id_1 < chunk_id_2", name="similarity_order_check"),
        sa.CheckConstraint("similarity >= 0.0 AND similarity <= 1.0", name="similarity_range_check"),
    )

    op.create_index("idx_chunk_similarities_1", "chunk_similarities", ["chunk_id_1"])
    op.create_index("idx_chunk_similarities_2", "chunk_similarities", ["chunk_id_2"])
    op.create_index("idx_chunk_similarities_sim", "chunk_similarities", [sa.text("similarity DESC")])

    # ==== REPO_EXPERIENCES TABLE ====
    op.create_table(
        "repo_experiences",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("repo_name", sa.Text(), nullable=False),
        sa.Column("experience_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("confidence", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("use_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("success_rate", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_repo_experiences")),
        sa.CheckConstraint(
            "experience_type IN ('pattern', 'pitfall', 'convention', 'solution')",
            name="experience_type_check"
        ),
    )

    op.create_index("idx_repo_experiences_repo", "repo_experiences", ["repo_name"])
    op.create_index("idx_repo_experiences_type", "repo_experiences", ["experience_type"])
    op.create_index("idx_repo_experiences_tags", "repo_experiences", ["tags"], postgresql_using="gin")
    
    # Vector index for semantic search
    op.create_index(
        "idx_repo_experiences_embedding",
        "repo_experiences",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Drop codebase index tables."""

    # Drop in reverse order of dependencies
    op.drop_index("idx_repo_experiences_embedding", table_name="repo_experiences")
    op.drop_index("idx_repo_experiences_tags", table_name="repo_experiences")
    op.drop_index("idx_repo_experiences_type", table_name="repo_experiences")
    op.drop_index("idx_repo_experiences_repo", table_name="repo_experiences")
    op.drop_table("repo_experiences")

    op.drop_index("idx_chunk_similarities_sim", table_name="chunk_similarities")
    op.drop_index("idx_chunk_similarities_2", table_name="chunk_similarities")
    op.drop_index("idx_chunk_similarities_1", table_name="chunk_similarities")
    op.drop_table("chunk_similarities")

    op.drop_index("idx_code_relations_type", table_name="code_relations")
    op.drop_index("idx_code_relations_to", table_name="code_relations")
    op.drop_index("idx_code_relations_from", table_name="code_relations")
    op.drop_table("code_relations")

    op.drop_index("idx_code_entities_chunk", table_name="code_entities")
    op.drop_index("idx_code_entities_type", table_name="code_entities")
    op.drop_index("idx_code_entities_name", table_name="code_entities")
    op.drop_index("idx_code_entities_repo", table_name="code_entities")
    op.drop_table("code_entities")

    op.execute("DROP INDEX IF EXISTS idx_code_chunks_search")
    op.execute("DROP INDEX IF EXISTS idx_code_chunks_hub_embedding")
    op.drop_index("idx_code_chunks_repo_file", table_name="code_chunks")
    op.drop_index("idx_code_chunks_file", table_name="code_chunks")
    op.drop_index("idx_code_chunks_repo", table_name="code_chunks")
    op.drop_table("code_chunks")
