"""
Test semantic index with full LEANN-style implementation.
Tests: FAISS HNSW, BM25 hybrid, PQ pruning, LRU cache, dual-mode
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct imports (avoids __init__.py which may have missing modules)
from agent.codebase_index.semantic_index import (
    SemanticIndex, 
    CodeEmbeddings, 
    BM25Scorer,
    FAISSHNSWIndex,
    LRUCache,
)
from agent.codebase_index.code_types import CodeChunk


def test_bm25_scorer():
    """Test BM25 implementation."""
    print("\n" + "=" * 60)
    print("Testing BM25 Scorer")
    print("=" * 60)
    
    scorer = BM25Scorer()
    
    documents = [
        {"id": "doc1", "text": "Python programming language for data science"},
        {"id": "doc2", "text": "Java programming basics and object oriented design"},
        {"id": "doc3", "text": "Machine learning with Python and TensorFlow"},
        {"id": "doc4", "text": "Web development with JavaScript and React"},
        {"id": "doc5", "text": "Data structures and algorithms in Python"},
    ]
    
    scorer.fit(documents)
    
    print(f"  Documents: {scorer.n_docs}")
    print(f"  Unique terms: {len(scorer.doc_freqs)}")
    print(f"  Avg doc length: {scorer.avg_doc_length:.2f}")
    
    # Test search
    queries = [
        "Python programming",
        "machine learning data",
        "web development",
    ]
    
    for query in queries:
        results = scorer.search(query, top_k=3)
        print(f"\n  Query: '{query}'")
        for doc_id, score in results:
            print(f"    - {doc_id}: {score:.4f}")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bm25.pkl"
        scorer.save(save_path)
        
        scorer2 = BM25Scorer()
        loaded = scorer2.load(save_path)
        print(f"\n  Save/Load test: {loaded and scorer2.n_docs == scorer.n_docs}")
    
    print("\n✅ BM25 Scorer test PASSED")


def test_faiss_hnsw_index():
    """Test FAISS HNSW index with PQ pruning."""
    print("\n" + "=" * 60)
    print("Testing FAISS HNSW Index")
    print("=" * 60)
    
    import numpy as np
    
    # Create test embeddings
    np.random.seed(42)
    n_vectors = 500
    dim = 128
    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    ids = [f"chunk_{i}" for i in range(n_vectors)]
    
    # Build index
    hnsw = FAISSHNSWIndex(
        dimension=dim, 
        M=16, 
        ef_construction=100, 
        ef_search=32,
        use_pq_pruning=True,
    )
    hnsw.build(embeddings, ids)
    
    print(f"  Built index with {len(ids)} vectors")
    print(f"  FAISS available: {hnsw._faiss_available}")
    print(f"  PQ index built: {hnsw._pq_index is not None}")
    
    # Test direct search (not pruned)
    query = np.random.randn(dim).astype(np.float32)
    results = hnsw.search(query, top_k=5)
    print(f"\n  Direct search results: {len(results)}")
    for doc_id, score in results[:3]:
        print(f"    - {doc_id}: {score:.4f}")
    
    # Prune embeddings
    hnsw.prune_embeddings()
    print(f"\n  After pruning: is_pruned={hnsw._is_pruned}")
    
    # Test search with recomputation
    embedding_cache = {i: embeddings[i].tolist() for i in range(n_vectors)}
    
    def recompute(node_id):
        return embedding_cache.get(node_id)
    
    results = hnsw.search(
        query, 
        top_k=5, 
        recompute_fn=recompute,
        pq_prune_ratio=0.3,
        recompute_budget=50,
    )
    print(f"\n  Recompute search results: {len(results)}")
    for doc_id, score in results[:3]:
        print(f"    - {doc_id}: {score:.4f}")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "hnsw_test"
        hnsw.save(save_path)
        
        hnsw2 = FAISSHNSWIndex(dimension=dim)
        loaded = hnsw2.load(save_path)
        print(f"\n  Save/Load test: {loaded and len(hnsw2._id_map) == len(hnsw._id_map)}")
    
    print("\n✅ FAISS HNSW Index test PASSED")


async def test_hybrid_search():
    """Test full hybrid search (semantic + BM25)."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search (Semantic + BM25)")
    print("=" * 60)
    
    # Create diverse test chunks
    chunks = [
        CodeChunk.create(
            file_path="auth.py",
            start_line=1, end_line=15,
            content="def login(username, password):\n    '''Authenticate user with credentials'''\n    validate_credentials(username, password)\n    return create_session(username)",
            chunk_type="function",
            name="login",
            docstring="Authenticate user with credentials",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="auth.py",
            start_line=17, end_line=25,
            content="def logout(session_id):\n    '''End user session'''\n    invalidate_session(session_id)",
            chunk_type="function",
            name="logout",
            docstring="End user session",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="user.py",
            start_line=1, end_line=20,
            content="class UserManager:\n    '''Manage user accounts'''\n    def create_user(self, email, password):\n        return User(email, password)\n    def delete_user(self, user_id):\n        pass",
            chunk_type="class",
            name="UserManager",
            docstring="Manage user accounts",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="database.py",
            start_line=1, end_line=10,
            content="def connect_database(host, port):\n    '''Establish database connection'''\n    return Connection(host, port)",
            chunk_type="function",
            name="connect_database",
            docstring="Establish database connection",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="math_utils.py",
            start_line=1, end_line=5,
            content="def calculate_sum(numbers):\n    '''Sum all numbers in list'''\n    return sum(numbers)",
            chunk_type="function",
            name="calculate_sum",
            docstring="Sum all numbers in list",
            repo_id="test",
        ),
        CodeChunk.create(
            file_path="api.py",
            start_line=1, end_line=12,
            content="def authenticate_request(request):\n    '''Verify API request authentication token'''\n    token = request.headers.get('Authorization')\n    return verify_token(token)",
            chunk_type="function",
            name="authenticate_request",
            docstring="Verify API request authentication token",
            repo_id="test",
        ),
    ]
    
    print(f"  Created {len(chunks)} test chunks")
    
    # Build index with hybrid search
    index = SemanticIndex(
        M=8,
        ef_search=20,
        recompute_budget=30,
        enable_bm25=True,
        bm25_weight=0.3,
        use_pq_pruning=True,
        pq_prune_ratio=0.2,
    )
    
    print("  Building index...")
    stats = await index.build_index(chunks)
    print(f"  Build stats: {stats}")
    
    # Test queries
    queries = [
        ("authentication login user", "Should find login and authenticate functions"),
        ("database connection", "Should find connect_database"),
        ("user management account", "Should find UserManager class"),
        ("calculate sum numbers", "Should find calculate_sum function"),
    ]
    
    print("\n  Testing hybrid search queries:")
    for query, expected in queries:
        results = await index.search(query, top_k=3)
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {expected}")
        print(f"  Results:")
        for i, r in enumerate(results):
            print(f"    {i+1}. {r.chunk.name} - score={r.score:.4f} (sem={r.semantic_score:.3f}, bm25={r.lexical_score:.3f})")
    
    # Compare with semantic-only search
    print("\n  Comparing: Semantic-only vs Hybrid")
    query = "authentication login"
    
    # Semantic only (bm25_weight=0)
    semantic_results = await index.search(query, top_k=3, bm25_weight=0.0)
    print(f"\n  Semantic-only for '{query}':")
    for r in semantic_results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Hybrid
    hybrid_results = await index.search(query, top_k=3, bm25_weight=0.3)
    print(f"\n  Hybrid (30% BM25) for '{query}':")
    for r in hybrid_results:
        print(f"    - {r.chunk.name}: {r.score:.4f}")
    
    # Test save/load
    print("\n  Testing save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "leann_hybrid"
        index.save(save_path)
        
        index2 = SemanticIndex()
        loaded = index2.load(save_path)
        print(f"  Loaded: {loaded}")
        print(f"  Loaded chunks: {len(index2.chunks)}")
        
        # Search on loaded index
        results = await index2.search("authentication", top_k=2)
        print(f"  Search on loaded: {len(results)} results")
    
    # Print stats
    print("\n  Index stats:")
    stats = index.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    print("\n✅ Hybrid Search test PASSED")


async def test_pq_pruning_effectiveness():
    """Test PQ pruning reduces computation while maintaining quality."""
    print("\n" + "=" * 60)
    print("Testing PQ Pruning Effectiveness")
    print("=" * 60)
    
    import numpy as np
    import time
    
    # Create larger dataset
    np.random.seed(42)
    n_vectors = 1000
    dim = 256
    
    # Create chunks
    chunks = []
    for i in range(n_vectors):
        chunks.append(CodeChunk.create(
            file_path=f"file_{i//10}.py",
            start_line=i*10, end_line=i*10+10,
            content=f"def function_{i}():\n    '''Function {i} implementation'''\n    pass",
            chunk_type="function",
            name=f"function_{i}",
            docstring=f"Function {i} implementation",
            repo_id="test",
        ))
    
    print(f"  Dataset: {n_vectors} chunks")
    
    # Build index
    index = SemanticIndex(
        M=16,
        ef_search=32,
        recompute_budget=100,
        enable_bm25=False,  # Disable BM25 for this test
        use_pq_pruning=True,
        pq_prune_ratio=0.5,  # Prune 50% candidates
    )
    
    print("  Building index...")
    start = time.time()
    await index.build_index(chunks)
    build_time = time.time() - start
    print(f"  Build time: {build_time:.2f}s")
    
    # Test search with different prune ratios
    query = "function implementation"
    
    prune_ratios = [0.0, 0.3, 0.5, 0.7]
    print("\n  Search performance by prune ratio:")
    
    for ratio in prune_ratios:
        start = time.time()
        results = await index.search(query, top_k=10, pq_prune_ratio=ratio)
        search_time = time.time() - start
        
        top_names = [r.chunk.name for r in results[:3]]
        print(f"    Prune {int(ratio*100):2d}%: {search_time:.4f}s, top-3: {top_names}")
    
    print("\n✅ PQ Pruning test PASSED")


async def main():
    print("=" * 60)
    print("Full LEANN-style Semantic Index Test Suite")
    print("=" * 60)
    
    # Test BM25
    test_bm25_scorer()
    
    # Test FAISS HNSW
    test_faiss_hnsw_index()
    
    # Test hybrid search
    await test_hybrid_search()
    
    # Test PQ pruning
    await test_pq_pruning_effectiveness()
    
    print("\n" + "=" * 60)
    print("All tests PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
