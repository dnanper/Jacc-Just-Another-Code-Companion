"""
Test structural index with knowledge graph.
Tests: Entity indexing, relations, spreading activation search, PageRank importance
"""

import sys
import tempfile
from pathlib import Path

# Add parent to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.codebase_index.structural_index import StructuralIndex
from agent.codebase_index.code_types import CodeChunk, CodeEntity, CodeRelation


def create_test_data():
    """Create test entities, relations, and chunks for a simple project."""
    
    # Create chunks (code units)
    chunks = [
        CodeChunk.create(
            file_path="models/user.py",
            start_line=1, end_line=50,
            content="class User:\n    def __init__(self, email):\n        self.email = email\n    def save(self):\n        db.save(self)\n    def validate(self):\n        pass",
            chunk_type="class",
            name="User",
            docstring="User model class",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="models/admin.py",
            start_line=1, end_line=30,
            content="class Admin(User):\n    def __init__(self, email, role):\n        super().__init__(email)\n        self.role = role\n    def has_permission(self, perm):\n        pass",
            chunk_type="class",
            name="Admin",
            docstring="Admin user class",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="services/auth.py",
            start_line=1, end_line=40,
            content="def authenticate(email, password):\n    user = User.query(email)\n    if user.validate():\n        return create_session(user)\n    return None",
            chunk_type="function",
            name="authenticate",
            docstring="Authenticate user",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="services/auth.py",
            start_line=42, end_line=50,
            content="def create_session(user):\n    return Session(user_id=user.id)",
            chunk_type="function",
            name="create_session",
            docstring="Create user session",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="api/views.py",
            start_line=1, end_line=30,
            content="def login_view(request):\n    result = authenticate(request.email, request.password)\n    return response(result)",
            chunk_type="function",
            name="login_view",
            docstring="Login API endpoint",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="api/views.py",
            start_line=32, end_line=50,
            content="def admin_view(request):\n    admin = Admin.query(request.user_id)\n    if admin.has_permission('admin'):\n        return admin_data()",
            chunk_type="function",
            name="admin_view",
            docstring="Admin API endpoint",
            repo_id="test",
        ),
    ]
    
    # Create entities linked to chunks
    entities = [
        CodeEntity.create("User", "class", "models/user.py", 1, chunk_id=chunks[0].id, repo_id="test"),
        CodeEntity.create("User.save", "method", "models/user.py", 5, chunk_id=chunks[0].id, repo_id="test"),
        CodeEntity.create("User.validate", "method", "models/user.py", 8, chunk_id=chunks[0].id, repo_id="test"),
        CodeEntity.create("Admin", "class", "models/admin.py", 1, chunk_id=chunks[1].id, repo_id="test"),
        CodeEntity.create("Admin.has_permission", "method", "models/admin.py", 6, chunk_id=chunks[1].id, repo_id="test"),
        CodeEntity.create("authenticate", "function", "services/auth.py", 1, chunk_id=chunks[2].id, repo_id="test"),
        CodeEntity.create("create_session", "function", "services/auth.py", 42, chunk_id=chunks[3].id, repo_id="test"),
        CodeEntity.create("login_view", "function", "api/views.py", 1, chunk_id=chunks[4].id, repo_id="test"),
        CodeEntity.create("admin_view", "function", "api/views.py", 32, chunk_id=chunks[5].id, repo_id="test"),
    ]
    
    # Create relations
    relations = [
        # Inheritance
        CodeRelation("Admin", "User", "inherits", weight=1.0),
        # Method calls
        CodeRelation("authenticate", "User.validate", "calls", weight=1.0),
        CodeRelation("authenticate", "create_session", "calls", weight=1.0),
        CodeRelation("login_view", "authenticate", "calls", weight=1.0),
        CodeRelation("admin_view", "Admin.has_permission", "calls", weight=1.0),
        # Uses
        CodeRelation("authenticate", "User", "uses", weight=0.8),
        CodeRelation("admin_view", "Admin", "uses", weight=0.8),
    ]
    
    return entities, relations, chunks


def test_build_index():
    """Test building structural index."""
    print("\n" + "=" * 60)
    print("Testing Build Index")
    print("=" * 60)
    
    entities, relations, chunks = create_test_data()
    
    index = StructuralIndex()
    stats = index.build_index(entities, relations, chunks)
    
    print(f"  Entities: {stats['entities']}")
    print(f"  Relations: {stats['relations']}")
    print(f"  Unresolved: {stats['unresolved']}")
    print(f"  Entity types: {stats['entity_types']}")
    
    assert stats['entities'] == 9
    assert stats['relations'] == 7
    
    print("\n✅ Build Index test PASSED")
    return index


def test_pagerank_importance(index: StructuralIndex):
    """Test PageRank importance scoring."""
    print("\n" + "=" * 60)
    print("Testing PageRank Importance")
    print("=" * 60)
    
    most_important = index.get_most_important(top_k=5)
    
    print("  Most important entities:")
    for entity, score in most_important:
        print(f"    - {entity.name}: {score:.4f}")
    
    # User and authenticate should be highly important (many callers)
    names = [e.name for e, _ in most_important]
    print(f"\n  Top 5 names: {names}")
    
    # User.validate and create_session should be important (called by authenticate)
    assert any("User" in n for n in names), "User-related entities should be important"
    
    print("\n✅ PageRank Importance test PASSED")


def test_spreading_activation_search(index: StructuralIndex):
    """Test spreading activation search."""
    print("\n" + "=" * 60)
    print("Testing Spreading Activation Search")
    print("=" * 60)
    
    # Search from login_view
    results = index.search(["login_view"], max_depth=2, top_k=5)
    
    print("  Search from 'login_view' (depth=2):")
    for r in results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Should find authenticate, User.validate, create_session
    result_names = [r.chunk.name for r in results]
    assert "authenticate" in result_names, "Should find authenticate"
    
    # Search for Admin-related
    results = index.search(["Admin"], max_depth=2, top_k=5)
    
    print("\n  Search from 'Admin' (depth=2):")
    for r in results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Should find User (parent class) and admin_view (uses Admin)
    result_names = [r.chunk.name for r in results]
    assert "User" in result_names, "Should find User (parent class)"
    
    print("\n✅ Spreading Activation Search test PASSED")


def test_relation_filtering(index: StructuralIndex):
    """Test filtering by relation type."""
    print("\n" + "=" * 60)
    print("Testing Relation Filtering")
    print("=" * 60)
    
    # Only inheritance
    results = index.search(["Admin"], relation_types=["inherits"], max_depth=1, top_k=5)
    
    print("  Search from 'Admin' (only 'inherits'):")
    for r in results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Should only find User (parent class)
    result_names = [r.chunk.name for r in results]
    assert "User" in result_names
    assert "admin_view" not in result_names  # Uses Admin, but we filtered to inherits only
    
    # Only calls
    results = index.search(["authenticate"], relation_types=["calls"], max_depth=1, top_k=5)
    
    print("\n  Search from 'authenticate' (only 'calls'):")
    for r in results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    print("\n✅ Relation Filtering test PASSED")


def test_helper_methods(index: StructuralIndex):
    """Test convenience helper methods."""
    print("\n" + "=" * 60)
    print("Testing Helper Methods")
    print("=" * 60)
    
    # get_callers
    callers = index.get_callers("authenticate")
    caller_names = [e.name for e in callers]
    print(f"  Callers of 'authenticate': {caller_names}")
    assert "login_view" in caller_names
    
    # get_callees
    callees = index.get_callees("authenticate")
    callee_names = [e.name for e in callees]
    print(f"  Callees of 'authenticate': {callee_names}")
    assert "User.validate" in callee_names
    assert "create_session" in callee_names
    
    # get_subclasses
    subclasses = index.get_subclasses("User")
    subclass_names = [e.name for e in subclasses]
    print(f"  Subclasses of 'User': {subclass_names}")
    assert "Admin" in subclass_names
    
    # get_base_classes
    bases = index.get_base_classes("Admin")
    base_names = [e.name for e in bases]
    print(f"  Base classes of 'Admin': {base_names}")
    assert "User" in base_names
    
    # get_users
    users = index.get_users("User")
    user_names = [e.name for e in users]
    print(f"  Users of 'User': {user_names}")
    assert "authenticate" in user_names
    
    print("\n✅ Helper Methods test PASSED")


def test_save_load():
    """Test save and load functionality."""
    print("\n" + "=" * 60)
    print("Testing Save/Load")
    print("=" * 60)
    
    entities, relations, chunks = create_test_data()
    
    # Build and save
    index = StructuralIndex()
    index.build_index(entities, relations, chunks)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "structural_test"
        index.save(save_path)
        
        print(f"  Saved to: {save_path}")
        print(f"  Files: {list(save_path.iterdir())}")
        
        # Load into new index
        index2 = StructuralIndex()
        loaded = index2.load(save_path)
        
        print(f"\n  Loaded: {loaded}")
        print(f"  Entities: {len(index2.entities)}")
        print(f"  Chunks: {len(index2.chunks)}")
        print(f"  Has importance: {len(index2.importance) > 0}")
        
        assert loaded
        assert len(index2.entities) == len(index.entities)
        
        # Test search on loaded index
        results = index2.search(["authenticate"], top_k=3)
        print(f"\n  Search on loaded index: {len(results)} results")
        for r in results:
            print(f"    - {r.chunk.name}")
    
    print("\n✅ Save/Load test PASSED")


def test_get_stats():
    """Test stats reporting."""
    print("\n" + "=" * 60)
    print("Testing Get Stats")
    print("=" * 60)
    
    entities, relations, chunks = create_test_data()
    
    index = StructuralIndex(
        decay_outgoing=0.85,
        decay_incoming=0.65,
        activation_threshold=0.05,
    )
    index.build_index(entities, relations, chunks)
    
    stats = index.get_stats()
    
    print("  Index stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    assert stats['initialized'] == True
    assert stats['entities'] == 9
    assert stats['decay_outgoing'] == 0.85
    assert stats['has_importance'] == True
    
    print("\n✅ Get Stats test PASSED")


def main():
    print("=" * 60)
    print("Structural Index Test Suite")
    print("=" * 60)
    
    # Test build
    index = test_build_index()
    
    # Test PageRank
    test_pagerank_importance(index)
    
    # Test search
    test_spreading_activation_search(index)
    
    # Test filtering
    test_relation_filtering(index)
    
    # Test helpers
    test_helper_methods(index)
    
    # Test save/load
    test_save_load()
    
    # Test stats
    test_get_stats()
    
    print("\n" + "=" * 60)
    print("All tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
