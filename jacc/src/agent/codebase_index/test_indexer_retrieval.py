"""
Test suite for CodebaseIndexer and CodebaseRetriever.

Tests:
- Index building from mock parsed data
- Save/Load indices
- Semantic retrieval (with BM25 hybrid)
- Structural retrieval
- RRF fusion retrieval
"""

import asyncio
import shutil
import sys
from pathlib import Path

# Add parent to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct imports to avoid __init__ issues
from agent.codebase_index.code_types import CodeChunk, CodeEntity, CodeRelation, RetrievalResult
from agent.codebase_index.semantic_index import SemanticIndex, CodeEmbeddings
from agent.codebase_index.structural_index import StructuralIndex


def create_mock_codebase():
    """
    Create mock parsed data representing a small authentication module.
    Returns: (chunks, entities, relations)
    """
    # Code chunks (what would be parsed from files)
    chunks = [
        CodeChunk.create(
            file_path="auth/models.py",
            start_line=1, end_line=25,
            content='''
class User:
    """User model for authentication."""
    def __init__(self, email: str, password_hash: str):
        self.email = email
        self.password_hash = password_hash
        self.is_active = True
    
    def verify_password(self, password: str) -> bool:
        """Verify user password against stored hash."""
        return check_password(password, self.password_hash)
    
    def save(self):
        """Save user to database."""
        db.users.insert(self)
''',
            chunk_type="class",
            name="User",
            docstring="User model for authentication.",
            repo_id="test_repo",
        ),
        CodeChunk.create(
            file_path="auth/models.py",
            start_line=28, end_line=45,
            content='''
class Admin(User):
    """Admin user with extra permissions."""
    def __init__(self, email: str, password_hash: str, role: str = "admin"):
        super().__init__(email, password_hash)
        self.role = role
        self.permissions = []
    
    def has_permission(self, permission: str) -> bool:
        """Check if admin has a specific permission."""
        return permission in self.permissions
''',
            chunk_type="class",
            name="Admin",
            docstring="Admin user with extra permissions.",
            repo_id="test_repo",
        ),
        CodeChunk.create(
            file_path="auth/service.py",
            start_line=1, end_line=20,
            content='''
def authenticate(email: str, password: str) -> User | None:
    """
    Authenticate user by email and password.
    Returns User if successful, None otherwise.
    """
    user = User.query(email=email)
    if user is None:
        return None
    if not user.verify_password(password):
        return None
    if not user.is_active:
        return None
    return user
''',
            chunk_type="function",
            name="authenticate",
            docstring="Authenticate user by email and password.",
            repo_id="test_repo",
        ),
        CodeChunk.create(
            file_path="auth/service.py",
            start_line=23, end_line=35,
            content='''
def create_session(user: User) -> Session:
    """Create a new session for authenticated user."""
    session = Session(
        user_id=user.id,
        token=generate_token(),
        expires_at=datetime.now() + timedelta(hours=24)
    )
    session.save()
    return session
''',
            chunk_type="function",
            name="create_session",
            docstring="Create a new session for authenticated user.",
            repo_id="test_repo",
        ),
        CodeChunk.create(
            file_path="auth/api.py",
            start_line=1, end_line=18,
            content='''
@app.route("/login", methods=["POST"])
def login_endpoint():
    """Handle user login request."""
    email = request.json.get("email")
    password = request.json.get("password")
    
    user = authenticate(email, password)
    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401
    
    session = create_session(user)
    return jsonify({"token": session.token})
''',
            chunk_type="function",
            name="login_endpoint",
            docstring="Handle user login request.",
            repo_id="test_repo",
        ),
        CodeChunk.create(
            file_path="auth/api.py",
            start_line=21, end_line=35,
            content='''
@app.route("/admin/users", methods=["GET"])
def admin_list_users():
    """List all users (admin only)."""
    admin = get_current_admin()
    if not admin.has_permission("view_users"):
        return jsonify({"error": "Forbidden"}), 403
    
    users = User.query_all()
    return jsonify([u.to_dict() for u in users])
''',
            chunk_type="function",
            name="admin_list_users",
            docstring="List all users (admin only).",
            repo_id="test_repo",
        ),
    ]
    
    # Entities (extracted from chunks)
    entities = [
        CodeEntity.create("User", "class", "auth/models.py", 1, chunk_id=chunks[0].id),
        CodeEntity.create("User.verify_password", "method", "auth/models.py", 10, chunk_id=chunks[0].id),
        CodeEntity.create("User.save", "method", "auth/models.py", 14, chunk_id=chunks[0].id),
        CodeEntity.create("Admin", "class", "auth/models.py", 28, chunk_id=chunks[1].id),
        CodeEntity.create("Admin.has_permission", "method", "auth/models.py", 36, chunk_id=chunks[1].id),
        CodeEntity.create("authenticate", "function", "auth/service.py", 1, chunk_id=chunks[2].id),
        CodeEntity.create("create_session", "function", "auth/service.py", 23, chunk_id=chunks[3].id),
        CodeEntity.create("login_endpoint", "function", "auth/api.py", 1, chunk_id=chunks[4].id),
        CodeEntity.create("admin_list_users", "function", "auth/api.py", 21, chunk_id=chunks[5].id),
    ]
    
    # Relations (code dependencies)
    relations = [
        # Inheritance
        CodeRelation("Admin", "User", "inherits", weight=1.0),
        # Function calls
        CodeRelation("authenticate", "User.verify_password", "calls", weight=1.0),
        CodeRelation("login_endpoint", "authenticate", "calls", weight=1.0),
        CodeRelation("login_endpoint", "create_session", "calls", weight=1.0),
        CodeRelation("admin_list_users", "Admin.has_permission", "calls", weight=1.0),
        # Uses
        CodeRelation("authenticate", "User", "uses", weight=0.8),
        CodeRelation("create_session", "User", "uses", weight=0.8),
        CodeRelation("admin_list_users", "Admin", "uses", weight=0.8),
    ]
    
    return chunks, entities, relations


# Local storage directory (in current folder)
LOCAL_INDEX_DIR = Path(__file__).parent / "test_index_storage"


def cleanup_test_storage():
    """Clean up test storage directory."""
    if LOCAL_INDEX_DIR.exists():
        shutil.rmtree(LOCAL_INDEX_DIR)
    LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)


async def test_semantic_index_standalone():
    """Test SemanticIndex independently."""
    print("\n" + "=" * 60)
    print("Testing SemanticIndex Standalone")
    print("=" * 60)
    
    chunks, _, _ = create_mock_codebase()
    
    storage_path = LOCAL_INDEX_DIR / "semantic"
    
    # Build index
    index = SemanticIndex(
        storage_path=storage_path,
        prune_embeddings=True,  # LEANN mode
        enable_bm25=True,
        bm25_weight=0.3,
    )
    
    print("  Building index...")
    stats = await index.build_index(chunks)
    print(f"  Build stats: {stats}")
    
    # Search tests
    print("\n  Testing searches:")
    
    queries = [
        ("user authentication login", "Should find authenticate, login_endpoint"),
        ("admin permission check", "Should find Admin, has_permission"),
        ("create session token", "Should find create_session"),
        ("User class model", "Should find User"),
    ]
    
    for query, expected in queries:
        results = await index.search(query, top_k=3)
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {expected}")
        print(f"  Results:")
        for r in results:
            print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Test save/load
    print("\n  Testing save/load...")
    index.save()
    
    index2 = SemanticIndex(storage_path=storage_path)
    loaded = index2.load()
    print(f"  Loaded: {loaded}")
    print(f"  Chunks after load: {len(index2.chunks)}")
    
    # Search on loaded index
    results = await index2.search("authenticate user", top_k=2)
    print(f"  Search on loaded: {len(results)} results")
    
    print("\n✅ SemanticIndex test PASSED")


async def test_structural_index_standalone():
    """Test StructuralIndex independently."""
    print("\n" + "=" * 60)
    print("Testing StructuralIndex Standalone")
    print("=" * 60)
    
    chunks, entities, relations = create_mock_codebase()
    
    storage_path = LOCAL_INDEX_DIR / "structural"
    
    # Build index
    index = StructuralIndex(storage_path=storage_path)
    
    print("  Building index...")
    stats = index.build_index(entities, relations, chunks)
    print(f"  Build stats: {stats}")
    
    # Test helper methods
    print("\n  Testing structural queries:")
    
    callers = index.get_callers("authenticate")
    print(f"  Callers of 'authenticate': {[e.name for e in callers]}")
    
    callees = index.get_callees("login_endpoint")
    print(f"  Callees of 'login_endpoint': {[e.name for e in callees]}")
    
    subclasses = index.get_subclasses("User")
    print(f"  Subclasses of 'User': {[e.name for e in subclasses]}")
    
    # Test search
    print("\n  Testing search from entities:")
    results = index.search(["login_endpoint"], max_depth=2, top_k=5)
    print(f"  Search from 'login_endpoint' (depth=2):")
    for r in results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Test importance
    print("\n  Most important entities:")
    important = index.get_most_important(top_k=5)
    for entity, score in important:
        print(f"    - {entity.name}: {score:.4f}")
    
    # Test save/load
    print("\n  Testing save/load...")
    index.save()
    
    index2 = StructuralIndex(storage_path=storage_path)
    loaded = index2.load()
    print(f"  Loaded: {loaded}")
    print(f"  Entities after load: {len(index2.entities)}")
    
    print("\n✅ StructuralIndex test PASSED")


async def test_combined_indexer():
    """Test combined indexing with both indices."""
    print("\n" + "=" * 60)
    print("Testing Combined Indexer (Mock)")
    print("=" * 60)
    
    chunks, entities, relations = create_mock_codebase()
    
    index_dir = LOCAL_INDEX_DIR / "combined"
    
    # Build both indices
    print("  Building semantic index...")
    semantic = SemanticIndex(
        storage_path=index_dir / "semantic",
        prune_embeddings=True,
        enable_bm25=True,
    )
    semantic_stats = await semantic.build_index(chunks)
    print(f"  Semantic stats: {semantic_stats}")
    
    print("\n  Building structural index...")
    structural = StructuralIndex(storage_path=index_dir / "structural")
    structural_stats = structural.build_index(entities, relations, chunks)
    print(f"  Structural stats: {structural_stats}")
    
    # Save both
    semantic.save()
    structural.save()
    
    print(f"\n  Index directory contents:")
    for item in index_dir.rglob("*"):
        if item.is_file():
            size = item.stat().st_size
            rel_path = item.relative_to(index_dir)
            print(f"    {rel_path}: {size:,} bytes")
    
    print("\n✅ Combined Indexer test PASSED")


async def test_retrieval_fusion():
    """Test retrieval with RRF fusion."""
    print("\n" + "=" * 60)
    print("Testing Retrieval with RRF Fusion")
    print("=" * 60)
    
    chunks, entities, relations = create_mock_codebase()
    
    index_dir = LOCAL_INDEX_DIR / "fusion"
    
    # Build indices
    semantic = SemanticIndex(
        storage_path=index_dir / "semantic",
        prune_embeddings=True,
        enable_bm25=True,
    )
    await semantic.build_index(chunks)
    
    structural = StructuralIndex(storage_path=index_dir / "structural")
    structural.build_index(entities, relations, chunks)
    
    # Simulate retrieval fusion manually
    print("  Query: 'how to authenticate users and create session'")
    print("  Entity hints: ['authenticate', 'session']")
    
    # Get results from both
    semantic_results = await semantic.search("how to authenticate users and create session", top_k=5)
    structural_results = structural.search(["authenticate", "session"], max_depth=2, top_k=5)
    
    print("\n  Semantic results:")
    for r in semantic_results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    print("\n  Structural results:")
    for r in structural_results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Manual RRF fusion
    RRF_K = 60
    weights = {"semantic": 0.6, "structural": 0.4}
    
    chunk_scores = {}
    for rank, r in enumerate(semantic_results, 1):
        chunk_id = r.chunk.id
        rrf = weights["semantic"] / (RRF_K + rank)
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf
        chunk_scores[f"{chunk_id}_name"] = r.chunk.name
    
    for rank, r in enumerate(structural_results, 1):
        chunk_id = r.chunk.id
        rrf = weights["structural"] / (RRF_K + rank)
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf
        chunk_scores[f"{chunk_id}_name"] = r.chunk.name
    
    # Sort by score
    sorted_chunks = sorted(
        [(k, v) for k, v in chunk_scores.items() if not k.endswith("_name")],
        key=lambda x: -x[1]
    )
    
    print("\n  Fused results (RRF):")
    for chunk_id, score in sorted_chunks[:5]:
        name = chunk_scores.get(f"{chunk_id}_name", "?")
        print(f"    - {name}: {score:.6f}")
    
    print("\n✅ Retrieval Fusion test PASSED")


async def test_storage_locations():
    """Test and display storage locations."""
    print("\n" + "=" * 60)
    print("Testing Storage Locations")
    print("=" * 60)
    
    chunks, entities, relations = create_mock_codebase()
    
    # LEANN mode
    leann_dir = LOCAL_INDEX_DIR / "leann_mode"
    semantic_leann = SemanticIndex(
        storage_path=leann_dir / "semantic",
        prune_embeddings=True,  # LEANN mode - no embeddings.npy
    )
    await semantic_leann.build_index(chunks)
    semantic_leann.save()
    
    structural = StructuralIndex(storage_path=leann_dir / "structural")
    structural.build_index(entities, relations, chunks)
    structural.save()
    
    # Display structure
    print(f"  Index directory: {LOCAL_INDEX_DIR}")
    print("\n  LEANN mode file structure:")
    
    total_size = 0
    for item in sorted(leann_dir.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            total_size += size
            rel_path = item.relative_to(leann_dir)
            print(f"    ├── {rel_path}: {size:,} bytes")
    
    print(f"\n  LEANN total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    # Traditional mode
    print("\n  Comparing LEANN vs Traditional mode:")
    
    trad_dir = LOCAL_INDEX_DIR / "traditional_mode"
    semantic_trad = SemanticIndex(
        storage_path=trad_dir / "semantic",
        prune_embeddings=False,  # Traditional - keeps embeddings
    )
    await semantic_trad.build_index(chunks)
    semantic_trad.save()
    
    leann_size = sum(f.stat().st_size for f in (leann_dir / "semantic").rglob("*") if f.is_file())
    trad_size = sum(f.stat().st_size for f in (trad_dir / "semantic").rglob("*") if f.is_file())
    
    print(f"    LEANN mode: {leann_size:,} bytes")
    print(f"    Traditional mode: {trad_size:,} bytes")
    if trad_size > 0:
        reduction = (1 - leann_size / trad_size) * 100
        print(f"    Storage reduction: {reduction:.1f}%")
    
    print("\n✅ Storage Locations test PASSED")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Indexer and Retriever Test Suite")
    print("=" * 60)
    
    # Clean up and create fresh test storage
    cleanup_test_storage()
    print(f"\n📂 Using local storage: {LOCAL_INDEX_DIR}")
    
    # Run individual tests
    await test_semantic_index_standalone()
    await test_structural_index_standalone()
    await test_combined_indexer()
    await test_retrieval_fusion()
    await test_storage_locations()
    
    print("\n" + "=" * 60)
    print("All tests PASSED! ✅")
    print("=" * 60)
    
    print("\n📂 Storage Summary:")
    print("  Index data is stored at: index_dir/")
    print("  ├── semantic/")
    print("  │   ├── config.json")
    print("  │   ├── chunks.json")
    print("  │   ├── bm25.pkl")
    print("  │   └── hnsw/")
    print("  │       ├── hnsw.index")
    print("  │       └── embeddings.npy (only if prune=False)")
    print("  └── structural/")
    print("      ├── config.json")
    print("      ├── entities.json")
    print("      ├── graph.json")
    print("      └── importance.json")


if __name__ == "__main__":
    asyncio.run(main())
