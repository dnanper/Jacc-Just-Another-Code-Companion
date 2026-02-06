"""
Test suite for TreeSitterParser.

Run with: python -m agent.codebase_index.test_code_parser
"""

import tempfile
from pathlib import Path

from code_parser import TreeSitterParser, ParseResult


# =============================================================================
# Sample Code for Testing
# =============================================================================

PYTHON_SAMPLE = '''
"""Module docstring."""

import os
from pathlib import Path
from typing import List, Optional

class BaseHandler:
    """Base class for handlers."""
    
    def handle(self, request):
        """Handle a request."""
        pass

class UserHandler(BaseHandler):
    """Handles user-related requests."""
    
    def __init__(self, db):
        self.db = db
    
    def get_user(self, user_id: int) -> Optional[dict]:
        """Get user by ID."""
        result = self.db.query(user_id)
        return result
    
    def create_user(self, name: str, email: str) -> dict:
        """Create a new user."""
        user = {"name": name, "email": email}
        self.db.save(user)
        return user

def process_request(handler: BaseHandler, data: dict) -> dict:
    """Process incoming request."""
    result = handler.handle(data)
    return result

async def async_handler(request):
    """Async request handler."""
    response = await fetch_data(request)
    return response
'''

JAVASCRIPT_SAMPLE = '''
import { Router } from 'express';
import UserService from './services/user';

class UserController {
    constructor(userService) {
        this.userService = userService;
    }
    
    async getUser(req, res) {
        const user = await this.userService.findById(req.params.id);
        res.json(user);
    }
    
    createUser(req, res) {
        const user = this.userService.create(req.body);
        res.status(201).json(user);
    }
}

function validateRequest(req) {
    return req.body && req.body.name;
}

const processData = (data) => {
    return data.map(item => item.value);
};

export default UserController;
'''

JAVA_SAMPLE = '''
package com.example.service;

import java.util.List;
import java.util.Optional;

public class UserService {
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
    
    public Optional<User> findById(Long id) {
        return repository.findById(id);
    }
    
    public List<User> findAll() {
        return repository.findAll();
    }
    
    public User create(User user) {
        return repository.save(user);
    }
}

interface UserRepository {
    Optional<User> findById(Long id);
    List<User> findAll();
    User save(User user);
}
'''

GO_SAMPLE = '''
package main

import (
    "fmt"
    "net/http"
)

type Handler struct {
    db *Database
}

func NewHandler(db *Database) *Handler {
    return &Handler{db: db}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, World!")
}

func main() {
    handler := NewHandler(nil)
    http.ListenAndServe(":8080", handler)
}
'''

RUST_SAMPLE = '''
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

enum Status {
    Active,
    Inactive,
    Pending,
}

impl User {
    fn new(id: u64, name: String, email: String) -> Self {
        User { id, name, email }
    }
    
    fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}

fn process_users(users: Vec<User>) -> HashMap<u64, User> {
    users.into_iter().map(|u| (u.id, u)).collect()
}
'''

C_SAMPLE = '''
#include <stdio.h>
#include <stdlib.h>

struct Point {
    int x;
    int y;
};

int add(int a, int b) {
    return a + b;
}

struct Point* create_point(int x, int y) {
    struct Point* p = malloc(sizeof(struct Point));
    p->x = x;
    p->y = y;
    return p;
}

int main() {
    int result = add(5, 3);
    printf("Result: %d\\n", result);
    return 0;
}
'''


# =============================================================================
# Test Functions
# =============================================================================

def test_python_parsing():
    """Test Python code parsing."""
    print("\n" + "="*60)
    print("Testing Python Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(PYTHON_SAMPLE)
        temp_path = Path(f.name)
    
    try:
        result = parser.parse_file(temp_path)
        _print_result(result, "Python")
        
        # Assertions
        assert len(result.chunks) >= 4, f"Expected at least 4 chunks, got {len(result.chunks)}"
        assert len(result.entities) >= 6, f"Expected at least 6 entities, got {len(result.entities)}"
        
        # Check specific extractions
        class_names = [c.name for c in result.chunks if c.chunk_type == "class"]
        assert "BaseHandler" in class_names, "BaseHandler class not found"
        assert "UserHandler" in class_names, "UserHandler class not found"
        
        func_names = [c.name for c in result.chunks if c.chunk_type == "function"]
        assert "process_request" in func_names, "process_request function not found"
        
        method_names = [c.name for c in result.chunks if c.chunk_type == "method"]
        assert any("get_user" in m for m in method_names), "get_user method not found"
        
        # Check imports
        import_entities = [e for e in result.entities if e.entity_type == "import"]
        assert len(import_entities) >= 2, f"Expected at least 2 imports, got {len(import_entities)}"
        
        # Check relations
        inheritance = [r for r in result.relations if r.relation_type == "inherits"]
        assert len(inheritance) >= 1, f"Expected at least 1 inheritance relation"
        
        print("✅ Python parsing test PASSED")
        return True
        
    except AssertionError as e:
        print(f"❌ Python parsing test FAILED: {e}")
        return False
    finally:
        temp_path.unlink()


def test_javascript_parsing():
    """Test JavaScript code parsing."""
    print("\n" + "="*60)
    print("Testing JavaScript Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False, encoding="utf-8") as f:
        f.write(JAVASCRIPT_SAMPLE)
        temp_path = Path(f.name)
    
    try:
        result = parser.parse_file(temp_path)
        _print_result(result, "JavaScript")
        
        # Assertions
        assert len(result.chunks) >= 2, f"Expected at least 2 chunks, got {len(result.chunks)}"
        
        # Check class
        class_chunks = [c for c in result.chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1, "Expected at least 1 class"
        
        # Check function
        func_chunks = [c for c in result.chunks if c.chunk_type == "function"]
        assert len(func_chunks) >= 1, "Expected at least 1 function"
        
        print("✅ JavaScript parsing test PASSED")
        return True
        
    except AssertionError as e:
        print(f"❌ JavaScript parsing test FAILED: {e}")
        return False
    finally:
        temp_path.unlink()


def test_java_parsing():
    """Test Java code parsing."""
    print("\n" + "="*60)
    print("Testing Java Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.NamedTemporaryFile(suffix=".java", mode="w", delete=False, encoding="utf-8") as f:
        f.write(JAVA_SAMPLE)
        temp_path = Path(f.name)
    
    try:
        result = parser.parse_file(temp_path)
        _print_result(result, "Java")
        
        # Assertions
        assert len(result.chunks) >= 2, f"Expected at least 2 chunks, got {len(result.chunks)}"
        
        # Check classes/interfaces
        class_chunks = [c for c in result.chunks if c.chunk_type == "class"]
        class_names = [c.name for c in class_chunks]
        assert "UserService" in class_names, "UserService not found"
        
        print("✅ Java parsing test PASSED")
        return True
        
    except AssertionError as e:
        print(f"❌ Java parsing test FAILED: {e}")
        return False
    finally:
        temp_path.unlink()


def test_go_parsing():
    """Test Go code parsing."""
    print("\n" + "="*60)
    print("Testing Go Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.NamedTemporaryFile(suffix=".go", mode="w", delete=False, encoding="utf-8") as f:
        f.write(GO_SAMPLE)
        temp_path = Path(f.name)
    
    try:
        result = parser.parse_file(temp_path)
        _print_result(result, "Go")
        
        # Assertions
        assert len(result.chunks) >= 2, f"Expected at least 2 chunks, got {len(result.chunks)}"
        
        # Check struct (treated as class)
        class_chunks = [c for c in result.chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1, "Expected at least 1 struct"
        
        # Check functions
        func_names = [c.name for c in result.chunks if c.chunk_type in ("function", "method")]
        assert len(func_names) >= 2, "Expected at least 2 functions/methods"
        
        print("✅ Go parsing test PASSED")
        return True
        
    except AssertionError as e:
        print(f"❌ Go parsing test FAILED: {e}")
        return False
    finally:
        temp_path.unlink()


def test_rust_parsing():
    """Test Rust code parsing."""
    print("\n" + "="*60)
    print("Testing Rust Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False, encoding="utf-8") as f:
        f.write(RUST_SAMPLE)
        temp_path = Path(f.name)
    
    try:
        result = parser.parse_file(temp_path)
        _print_result(result, "Rust")
        
        # Assertions
        assert len(result.chunks) >= 2, f"Expected at least 2 chunks, got {len(result.chunks)}"
        
        # Check struct
        class_chunks = [c for c in result.chunks if c.chunk_type == "class"]
        class_names = [c.name for c in class_chunks]
        assert "User" in class_names or "Status" in class_names, "User or Status not found"
        
        print("✅ Rust parsing test PASSED")
        return True
        
    except AssertionError as e:
        print(f"❌ Rust parsing test FAILED: {e}")
        return False
    finally:
        temp_path.unlink()


def test_c_parsing():
    """Test C code parsing."""
    print("\n" + "="*60)
    print("Testing C Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False, encoding="utf-8") as f:
        f.write(C_SAMPLE)
        temp_path = Path(f.name)
    
    try:
        result = parser.parse_file(temp_path)
        _print_result(result, "C")
        
        # Assertions
        assert len(result.chunks) >= 2, f"Expected at least 2 chunks, got {len(result.chunks)}"
        
        # Check functions
        func_chunks = [c for c in result.chunks if c.chunk_type == "function"]
        func_names = [c.name for c in func_chunks]
        assert "add" in func_names or "main" in func_names, "add or main function not found"
        
        print("✅ C parsing test PASSED")
        return True
        
    except AssertionError as e:
        print(f"❌ C parsing test FAILED: {e}")
        return False
    finally:
        temp_path.unlink()


def test_directory_parsing():
    """Test parsing a directory with multiple files."""
    print("\n" + "="*60)
    print("Testing Directory Parsing")
    print("="*60)
    
    parser = TreeSitterParser(repo_id="test-repo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create multiple files
        (tmpdir / "main.py").write_text(PYTHON_SAMPLE, encoding="utf-8")
        (tmpdir / "app.js").write_text(JAVASCRIPT_SAMPLE, encoding="utf-8")
        (tmpdir / "utils.go").write_text(GO_SAMPLE, encoding="utf-8")
        
        # Parse directory
        results = list(parser.parse_directory(tmpdir))
        
        print(f"\nParsed {len(results)} files")
        for file_path, result in results:
            print(f"  - {file_path.name}: {len(result.chunks)} chunks, {len(result.entities)} entities")
        
        # Check stats
        stats = parser.stats
        print(f"\nParser stats: {stats}")
        
        assert stats["files"] >= 3, f"Expected at least 3 files, got {stats['files']}"
        assert stats["chunks"] >= 6, f"Expected at least 6 chunks, got {stats['chunks']}"
        
        print("✅ Directory parsing test PASSED")
        return True


def _print_result(result: ParseResult, language: str):
    """Helper to print parse results."""
    print(f"\n{language} Parse Results:")
    print(f"  Chunks: {len(result.chunks)}")
    print(f"  Entities: {len(result.entities)}")
    print(f"  Relations: {len(result.relations)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.errors:
        print(f"  Error details: {result.errors}")
    
    print("\n  Chunks:")
    for chunk in result.chunks[:10]:  # Limit output
        print(f"    - [{chunk.chunk_type}] {chunk.name} (lines {chunk.start_line}-{chunk.end_line})")
    
    print("\n  Entities:")
    for entity in result.entities[:10]:
        print(f"    - [{entity.entity_type}] {entity.name}")
    
    if result.relations:
        print("\n  Relations:")
        for rel in result.relations[:5]:
            print(f"    - {rel.from_entity_id[:8]}... --[{rel.relation_type}]--> {rel.to_entity_id[:20]}...")


def run_all_tests():
    """Run all parser tests."""
    print("\n" + "="*60)
    print("TreeSitterParser Test Suite")
    print("="*60)
    
    # Check available languages
    parser = TreeSitterParser()
    print(f"\nSupported languages: {parser.supported_languages}")
    
    tests = [
        ("Python", test_python_parsing),
        ("JavaScript", test_javascript_parsing),
        ("Java", test_java_parsing),
        ("Go", test_go_parsing),
        ("Rust", test_rust_parsing),
        ("C", test_c_parsing),
        ("Directory", test_directory_parsing),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test ERROR: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
