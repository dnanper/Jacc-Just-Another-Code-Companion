"""
Multi-language code parser using tree-sitter with query-based extraction.

Uses tree-sitter's query language for declarative AST pattern matching,
making extraction consistent and maintainable across languages.

Supports: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++

Dependencies:
    pip install tree-sitter tree-sitter-python tree-sitter-javascript \
        tree-sitter-typescript tree-sitter-java tree-sitter-go \
        tree-sitter-rust tree-sitter-c
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_java as tsjava
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node, Query, QueryCursor

from code_types import CodeChunk, CodeEntity, CodeRelation

logger = logging.getLogger(__name__)


# =============================================================================
# Language Configuration
# =============================================================================

EXTENSION_TO_LANGUAGE = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
}

SKIP_PATTERNS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules", "venv", ".venv",
    "build", "dist", ".tox", ".pytest_cache", ".mypy_cache", "*.egg-info",
}


# =============================================================================
# Tree-Sitter Queries per Language
# =============================================================================
# Tree-sitter query syntax: (node_type field: (child_type) @capture_name)
# Each language has different AST node types, so queries are language-specific.
# But the EXTRACTION LOGIC remains unified via the QueryExtractor class.

LANGUAGE_QUERIES = {
    "python": """
        ; Classes
        (class_definition
            name: (identifier) @class.name
            superclasses: (argument_list)? @class.bases
            body: (block) @class.body) @class.def
        
        ; Functions
        (function_definition
            name: (identifier) @function.name
            parameters: (parameters) @function.params
            body: (block) @function.body) @function.def
        
        ; Decorated definitions
        (decorated_definition
            definition: (class_definition) @decorated.class)
        (decorated_definition
            definition: (function_definition) @decorated.function)
        
        ; Imports
        (import_statement
            name: (dotted_name) @import.name) @import.def
        (import_from_statement
            module_name: (dotted_name)? @import.module
            name: (dotted_name) @import.name) @import.from
        
        ; Function calls (for relations)
        (call
            function: (identifier) @call.name)
        (call
            function: (attribute) @call.attr)
    """,
    
    "javascript": """
        ; Classes
        (class_declaration
            name: (identifier) @class.name
            body: (class_body) @class.body) @class.def
        
        ; Functions
        (function_declaration
            name: (identifier) @function.name
            parameters: (formal_parameters) @function.params
            body: (statement_block) @function.body) @function.def
        
        ; Arrow functions in variable declarations
        (variable_declarator
            name: (identifier) @arrow.name
            value: (arrow_function) @arrow.func)
        
        ; Methods
        (method_definition
            name: (property_identifier) @method.name
            parameters: (formal_parameters) @method.params
            body: (statement_block) @method.body) @method.def
        
        ; Imports
        (import_statement
            source: (string) @import.source) @import.def
    """,
    
    "java": """
        ; Classes
        (class_declaration
            name: (identifier) @class.name
            superclass: (superclass)? @class.extends
            body: (class_body) @class.body) @class.def
        
        ; Interfaces
        (interface_declaration
            name: (identifier) @interface.name
            body: (interface_body) @interface.body) @interface.def
        
        ; Methods
        (method_declaration
            name: (identifier) @method.name
            parameters: (formal_parameters) @method.params
            body: (block)? @method.body) @method.def
        
        ; Imports
        (import_declaration
            (scoped_identifier) @import.name) @import.def
    """,
    
    "go": """
        ; Functions
        (function_declaration
            name: (identifier) @function.name
            parameters: (parameter_list) @function.params
            body: (block) @function.body) @function.def
        
        ; Methods
        (method_declaration
            receiver: (parameter_list) @method.receiver
            name: (field_identifier) @method.name
            parameters: (parameter_list) @method.params
            body: (block) @method.body) @method.def
        
        ; Types (struct, interface)
        (type_declaration
            (type_spec
                name: (type_identifier) @type.name
                type: (_) @type.body)) @type.def
        
        ; Imports
        (import_declaration
            (import_spec
                path: (interpreted_string_literal) @import.path)) @import.def
    """,
    
    "rust": """
        ; Functions
        (function_item
            name: (identifier) @function.name
            parameters: (parameters) @function.params
            body: (block) @function.body) @function.def
        
        ; Structs
        (struct_item
            name: (type_identifier) @struct.name) @struct.def
        
        ; Enums
        (enum_item
            name: (type_identifier) @enum.name) @enum.def
        
        ; Impl blocks
        (impl_item
            type: (type_identifier) @impl.type
            body: (declaration_list) @impl.body) @impl.def
        
        ; Use declarations
        (use_declaration
            argument: (_) @use.path) @use.def
    """,
    
    "c": """
        ; Functions
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @function.name)
            body: (compound_statement) @function.body) @function.def
        
        ; Pointer function (common pattern)
        (function_definition
            declarator: (pointer_declarator
                declarator: (function_declarator
                    declarator: (identifier) @function.name))
            body: (compound_statement) @function.body) @function.ptr_def
        
        ; Structs
        (struct_specifier
            name: (type_identifier) @struct.name
            body: (field_declaration_list)? @struct.body) @struct.def
        
        ; Includes
        (preproc_include
            path: (_) @include.path) @include.def
    """,
}

# TypeScript reuses JavaScript queries with additions
LANGUAGE_QUERIES["typescript"] = LANGUAGE_QUERIES["javascript"] + """
    ; TypeScript interfaces
    (interface_declaration
        name: (type_identifier) @interface.name
        body: (object_type) @interface.body) @interface.def
    
    ; Type aliases  
    (type_alias_declaration
        name: (type_identifier) @type.name
        value: (_) @type.value) @type.def
"""

# C++ reuses C queries with additions
LANGUAGE_QUERIES["cpp"] = LANGUAGE_QUERIES["c"] + """
    ; Classes
    (class_specifier
        name: (type_identifier) @class.name
        body: (field_declaration_list)? @class.body) @class.def
    
    ; Namespaces
    (namespace_definition
        name: (identifier)? @namespace.name
        body: (declaration_list) @namespace.body) @namespace.def
"""


# =============================================================================
# Parse Result
# =============================================================================

@dataclass
class ParseResult:
    """Result from parsing a single file."""
    chunks: list[CodeChunk] = field(default_factory=list)
    entities: list[CodeEntity] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Query-based Extractor
# =============================================================================

class QueryExtractor:
    """
    Unified extraction logic using tree-sitter query results.
    
    Each query capture is processed uniformly regardless of language.
    The capture name (e.g., "class.name", "function.def") determines
    how the node is processed.
    """
    
    def __init__(self, file_path: str, lines: list[str], repo_id: str | None = None):
        self.file_path = file_path
        self.lines = lines
        self.repo_id = repo_id
        self.result = ParseResult()
        
        # Track context for methods
        self._class_stack: list[tuple[str, CodeEntity]] = []  # [(class_name, entity), ...]
        self._current_class: CodeEntity | None = None
    
    def extract(self, root: Node, language: Language, query_text: str) -> ParseResult:
        """Run query and process all captures using matches()."""
        try:
            query = Query(language, query_text)
            query_cursor = QueryCursor(query)
        except Exception as e:
            self.result.errors.append(f"Query error: {e}")
            return self.result
        
        # Use matches() which returns list of (pattern_id, captures_dict)
        # Each match has captures grouped together
        matches = query_cursor.matches(root)
        
        for pattern_id, captures in matches:
            # captures is dict like {'func.name': [node], 'func.def': [node]}
            # Extract first node from each capture (usually only one)
            info = {}
            for capture_name, nodes in captures.items():
                if nodes:
                    info[capture_name] = nodes[0]  # Take first node
            
            # Process based on capture names present
            if 'class.name' in info:
                self._process_class(info)
            elif 'interface.name' in info:
                self._process_interface(info)
            elif 'function.name' in info:
                self._process_function(info)
            elif 'method.name' in info:
                self._process_method(info)
            elif 'arrow.name' in info:
                self._process_arrow_function(info)
            elif 'import.name' in info or 'import.source' in info or 'import.path' in info:
                self._process_import(info)
            elif 'type.name' in info or 'struct.name' in info:
                self._process_type(info)
            elif 'enum.name' in info:
                self._process_enum(info)
            elif 'impl.type' in info:
                self._process_impl(info)
            elif 'use.path' in info:
                self._process_use(info)
            elif 'include.path' in info:
                self._process_include(info)
            elif 'decorated.class' in info:
                # Handle decorated class - extract from the definition
                class_node = info['decorated.class']
                self._process_decorated_class(class_node)
            elif 'decorated.function' in info:
                # Handle decorated function
                func_node = info['decorated.function']
                self._process_decorated_function(func_node)
        
        return self.result
    
    def _node_text(self, node: Node | None) -> str:
        """Get text content of a node."""
        if node is None:
            return ""
        return node.text.decode("utf-8", errors="ignore") if node.text else ""
    
    def _get_content(self, node: Node) -> str:
        """Get source content for a node."""
        start = node.start_point[0]
        end = node.end_point[0]
        return "\n".join(self.lines[start:end + 1])
    
    def _get_docstring(self, body_node: Node | None) -> str | None:
        """Extract docstring from function/class body."""
        if not body_node:
            return None
        
        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        text = self._node_text(sub)
                        if text.startswith(('"""', "'''")):
                            return text[3:-3].strip()
                        elif text.startswith(('"', "'")):
                            return text[1:-1].strip()
            elif child.type not in ("comment", "NEWLINE", "{", "}"):
                break
        return None
    
    def _process_class(self, info: dict):
        """Process class definition."""
        name_node = info.get('class.name')
        def_node = info.get('class.def')
        body_node = info.get('class.body')
        bases_node = info.get('class.bases')
        
        if not name_node:
            return
        
        # If no def_node, try to find parent class_definition
        if not def_node:
            def_node = name_node
            while def_node and def_node.type != 'class_definition':
                def_node = def_node.parent
        
        if not def_node:
            return
        
        class_name = self._node_text(name_node)
        content = self._get_content(def_node)
        docstring = self._get_docstring(body_node)
        
        # Parse bases
        bases = []
        if bases_node:
            for child in bases_node.children:
                if child.type in ('identifier', 'attribute'):
                    bases.append(self._node_text(child))
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            content=content,
            chunk_type="class",
            name=class_name,
            docstring=docstring,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=class_name,
            entity_type="class",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            metadata={"bases": bases},
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
        
        # Track class context via stack
        self._class_stack.append((class_name, entity))
        
        # Inheritance relations
        for base in bases:
            if base not in ("object", "ABC", "Exception"):
                self.result.relations.append(CodeRelation(
                    from_entity_id=entity.id,
                    to_entity_id=base,
                    relation_type="inherits",
                ))
    
    def _process_interface(self, info: dict):
        """Process interface definition (Java/TypeScript)."""
        name_node = info.get('interface.name')
        def_node = info.get('interface.def')
        
        if not name_node or not def_node:
            return
        
        name = self._node_text(name_node)
        content = self._get_content(def_node)
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            content=content,
            chunk_type="class",
            name=name,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=name,
            entity_type="class",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            metadata={"is_interface": True},
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_function(self, info: dict):
        """Process function definition."""
        name_node = info.get('function.name')
        def_node = info.get('function.def') or info.get('function.ptr_def')
        body_node = info.get('function.body')
        params_node = info.get('function.params')
        
        if not name_node or not def_node:
            return
        
        func_name = self._node_text(name_node)
        content = self._get_content(def_node)
        docstring = self._get_docstring(body_node)
        
        # Check if this function is actually a method (inside a class)
        is_method = False
        class_name = None
        parent = def_node.parent
        while parent:
            if parent.type == 'class_definition':
                # Find class name
                for child in parent.children:
                    if child.type == 'identifier':
                        class_name = self._node_text(child)
                        break
                is_method = True
                break
            elif parent.type == 'module':
                break
            parent = parent.parent
        
        # Full name includes class if it's a method
        full_name = f"{class_name}.{func_name}" if is_method and class_name else func_name
        
        # Parse parameters
        params = []
        if params_node:
            for child in params_node.children:
                if child.type == 'identifier':
                    params.append(self._node_text(child))
                elif child.type in ('typed_parameter', 'parameter', 'formal_parameter'):
                    for sub in child.children:
                        if sub.type == 'identifier':
                            params.append(self._node_text(sub))
                            break
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            content=content,
            chunk_type="method" if is_method else "function",
            name=full_name,
            docstring=docstring,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=full_name,
            entity_type="method" if is_method else "function",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            metadata={"params": params},
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_method(self, info: dict):
        """Process method definition."""
        name_node = info.get('method.name')
        def_node = info.get('method.def')
        body_node = info.get('method.body')
        receiver_node = info.get('method.receiver')  # Go
        
        if not name_node or not def_node:
            return
        
        method_name = self._node_text(name_node)
        content = self._get_content(def_node)
        docstring = self._get_docstring(body_node)
        
        # Get class context
        class_context = None
        if self._class_stack:
            class_context = self._class_stack[-1]
        
        # For Go methods, extract receiver type
        if receiver_node and not class_context:
            receiver_text = self._node_text(receiver_node)
            # Extract type from receiver like "(r *Router)"
            parts = receiver_text.strip("()").split()
            if len(parts) >= 2:
                receiver_type = parts[-1].lstrip("*")
                full_name = f"{receiver_type}.{method_name}"
            else:
                full_name = method_name
        elif class_context:
            full_name = f"{class_context[0]}.{method_name}"
        else:
            full_name = method_name
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            content=content,
            chunk_type="method",
            name=full_name,
            docstring=docstring,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=full_name,
            entity_type="method",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
        
        # Create defines relation
        if class_context:
            self.result.relations.append(CodeRelation(
                from_entity_id=class_context[1].id,
                to_entity_id=entity.id,
                relation_type="defines",
            ))
    
    def _process_arrow_function(self, info: dict):
        """Process arrow function (JavaScript)."""
        name_node = info.get('arrow.name')
        func_node = info.get('arrow.func')
        
        if not name_node or not func_node:
            return
        
        func_name = self._node_text(name_node)
        content = self._get_content(func_node)
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            content=content,
            chunk_type="function",
            name=func_name,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=func_name,
            entity_type="function",
            file_path=self.file_path,
            line_number=func_node.start_point[0] + 1,
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_import(self, info: dict):
        """Process import statement."""
        def_node = info.get('import.def') or info.get('import.from')
        name_node = info.get('import.name')
        module_node = info.get('import.module')
        source_node = info.get('import.source')
        path_node = info.get('import.path')  # Go imports
        
        # Get line number from whichever node is available
        ref_node = def_node or name_node or source_node or path_node
        if not ref_node:
            return
        
        line = ref_node.start_point[0] + 1
        
        if source_node:
            # JavaScript import
            source = self._node_text(source_node).strip("'\"")
            entity = CodeEntity.create(
                name=source,
                entity_type="import",
                file_path=self.file_path,
                line_number=line,
                repo_id=self.repo_id,
            )
            self.result.entities.append(entity)
        elif path_node:
            # Go import
            path = self._node_text(path_node).strip('"')
            entity = CodeEntity.create(
                name=path,
                entity_type="import",
                file_path=self.file_path,
                line_number=line,
                repo_id=self.repo_id,
            )
            self.result.entities.append(entity)
        elif name_node:
            # Python import
            name = self._node_text(name_node)
            module = self._node_text(module_node) if module_node else ""
            full_name = f"{module}.{name}" if module else name
            
            entity = CodeEntity.create(
                name=full_name,
                entity_type="import",
                file_path=self.file_path,
                line_number=line,
                metadata={"module": module} if module else {},
                repo_id=self.repo_id,
            )
            self.result.entities.append(entity)
    
    def _process_type(self, info: dict):
        """Process type/struct definition."""
        name_node = info.get('type.name') or info.get('struct.name')
        def_node = info.get('type.def') or info.get('struct.def')
        
        if not name_node or not def_node:
            return
        
        type_name = self._node_text(name_node)
        content = self._get_content(def_node)
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            content=content,
            chunk_type="class",
            name=type_name,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=type_name,
            entity_type="class",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_enum(self, info: dict):
        """Process enum definition (Rust)."""
        name_node = info.get('enum.name')
        def_node = info.get('enum.def')
        
        if not name_node or not def_node:
            return
        
        enum_name = self._node_text(name_node)
        content = self._get_content(def_node)
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=def_node.start_point[0] + 1,
            end_line=def_node.end_point[0] + 1,
            content=content,
            chunk_type="class",
            name=enum_name,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=enum_name,
            entity_type="class",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            metadata={"is_enum": True},
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_impl(self, info: dict):
        """Process impl block (Rust) - functions inside are methods."""
        type_node = info.get('impl.type')
        body_node = info.get('impl.body')
        
        if type_node:
            impl_type = self._node_text(type_node)
            # Push impl type as class context
            dummy_entity = CodeEntity.create(
                name=impl_type,
                entity_type="class",
                file_path=self.file_path,
                line_number=0,
                repo_id=self.repo_id,
            )
            self._class_stack.append((impl_type, dummy_entity))
    
    def _process_use(self, info: dict):
        """Process use declaration (Rust)."""
        path_node = info.get('use.path')
        def_node = info.get('use.def')
        
        if not def_node:
            return
        
        path = self._node_text(path_node) if path_node else self._node_text(def_node)
        
        entity = CodeEntity.create(
            name=path,
            entity_type="import",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_include(self, info: dict):
        """Process #include (C/C++)."""
        path_node = info.get('include.path')
        def_node = info.get('include.def')
        
        if not def_node:
            return
        
        path = self._node_text(path_node).strip('"<>') if path_node else ""
        
        entity = CodeEntity.create(
            name=path,
            entity_type="import",
            file_path=self.file_path,
            line_number=def_node.start_point[0] + 1,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
    
    def _process_decorated_class(self, class_node: Node):
        """Process a decorated class definition."""
        # Find nested class name and body
        name_node = None
        body_node = None
        bases_node = None
        
        for child in class_node.children:
            if child.type == 'identifier' and not name_node:
                name_node = child
            elif child.type == 'block':
                body_node = child
            elif child.type == 'argument_list':
                bases_node = child
        
        if not name_node:
            return
        
        class_name = self._node_text(name_node)
        content = self._get_content(class_node)
        docstring = self._get_docstring(body_node)
        
        # Parse bases
        bases = []
        if bases_node:
            for child in bases_node.children:
                if child.type in ('identifier', 'attribute'):
                    bases.append(self._node_text(child))
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1,
            content=content,
            chunk_type="class",
            name=class_name,
            docstring=docstring,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=class_name,
            entity_type="class",
            file_path=self.file_path,
            line_number=class_node.start_point[0] + 1,
            metadata={"bases": bases},
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
        
        # Track class for methods
        self._class_stack.append((class_name, entity))
        
        # Inheritance relations
        for base in bases:
            if base not in ("object", "ABC", "Exception"):
                self.result.relations.append(CodeRelation(
                    from_entity_id=entity.id,
                    to_entity_id=base,
                    relation_type="inherits",
                ))
    
    def _process_decorated_function(self, func_node: Node):
        """Process a decorated function definition."""
        # Find nested function name and body
        name_node = None
        body_node = None
        params_node = None
        
        for child in func_node.children:
            if child.type == 'identifier' and not name_node:
                name_node = child
            elif child.type == 'block':
                body_node = child
            elif child.type == 'parameters':
                params_node = child
        
        if not name_node:
            return
        
        func_name = self._node_text(name_node)
        content = self._get_content(func_node)
        docstring = self._get_docstring(body_node)
        
        # Check if this is a method (inside a class)
        is_method = len(self._class_stack) > 0
        if is_method:
            full_name = f"{self._class_stack[-1][0]}.{func_name}"
        else:
            full_name = func_name
        
        # Parse parameters
        params = []
        if params_node:
            for child in params_node.children:
                if child.type == 'identifier':
                    params.append(self._node_text(child))
                elif child.type in ('typed_parameter', 'default_parameter'):
                    for sub in child.children:
                        if sub.type == 'identifier':
                            params.append(self._node_text(sub))
                            break
        
        chunk = CodeChunk.create(
            file_path=self.file_path,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            content=content,
            chunk_type="method" if is_method else "function",
            name=full_name,
            docstring=docstring,
            repo_id=self.repo_id,
        )
        self.result.chunks.append(chunk)
        
        entity = CodeEntity.create(
            name=full_name,
            entity_type="method" if is_method else "function",
            file_path=self.file_path,
            line_number=func_node.start_point[0] + 1,
            metadata={"params": params},
            chunk_id=chunk.id,
            repo_id=self.repo_id,
        )
        self.result.entities.append(entity)
        
        # Create defines relation for methods
        if is_method:
            self.result.relations.append(CodeRelation(
                from_entity_id=self._class_stack[-1][1].id,
                to_entity_id=entity.id,
                relation_type="defines",
            ))


# =============================================================================
# Main Parser Class
# =============================================================================

class TreeSitterParser:
    """
    Multi-language code parser using tree-sitter.
    
    Uses declarative queries for extraction across all supported languages.
    """
    
    def __init__(self, repo_id: str | None = None):
        self.repo_id = repo_id
        self._stats = {"files": 0, "chunks": 0, "entities": 0, "relations": 0, "errors": 0}
        
        # Initialize parsers and languages
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        self._init_languages()
    
    def _init_languages(self):
        """Initialize tree-sitter language parsers."""
        language_modules = {
            "python": tspython,
            "javascript": tsjavascript,
            "java": tsjava,
            "go": tsgo,
            "rust": tsrust,
            "c": tsc,
            "cpp": tsc,
        }
        
        for lang_name, module in language_modules.items():
            try:
                language = Language(module.language())
                self._languages[lang_name] = language
                parser = Parser(language)
                self._parsers[lang_name] = parser
            except Exception as e:
                logger.warning(f"Failed to init {lang_name}: {e}")
        
        # TypeScript uses JavaScript parser
        if "javascript" in self._languages:
            self._languages["typescript"] = self._languages["javascript"]
            self._parsers["typescript"] = self._parsers["javascript"]
    
    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()
    
    @property
    def supported_languages(self) -> set[str]:
        return set(self._parsers.keys())
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Parse a file using tree-sitter and query-based extraction."""
        result = ParseResult()
        
        if not file_path.exists() or not file_path.is_file():
            return result
        
        # Detect language
        ext = file_path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext)
        
        if not language or language not in self._parsers:
            return result
        
        # Read file
        try:
            content = file_path.read_bytes()
            content_str = content.decode("utf-8", errors="ignore")
            lines = content_str.split("\n")
        except Exception as e:
            result.errors.append(f"Read error: {e}")
            return result
        
        # Parse with tree-sitter
        parser = self._parsers[language]
        try:
            tree = parser.parse(content)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result
        
        # Get query for this language
        query_text = LANGUAGE_QUERIES.get(language, "")
        if not query_text:
            result.errors.append(f"No query for language: {language}")
            return result
        
        # Extract using query-based extractor
        lang_obj = self._languages[language]
        extractor = QueryExtractor(str(file_path), lines, self.repo_id)
        
        return extractor.extract(tree.root_node, lang_obj, query_text)
    
    def parse_directory(
        self,
        directory: Path,
        extensions: set[str] | None = None,
        max_files: int | None = None,
    ) -> Iterator[tuple[Path, ParseResult]]:
        """Parse all code files in a directory."""
        if not directory.exists() or not directory.is_dir():
            return
        
        files_processed = 0
        
        for file_path in self._iter_files(directory, extensions):
            if max_files and files_processed >= max_files:
                break
            
            result = self.parse_file(file_path)
            
            if result.chunks or result.entities:
                self._stats["files"] += 1
                self._stats["chunks"] += len(result.chunks)
                self._stats["entities"] += len(result.entities)
                self._stats["relations"] += len(result.relations)
                self._stats["errors"] += len(result.errors)
                files_processed += 1
                yield file_path, result
    
    def _iter_files(
        self,
        directory: Path,
        extensions: set[str] | None = None,
    ) -> Iterator[Path]:
        """Iterate over code files, skipping ignored patterns."""
        try:
            items = list(directory.iterdir())
        except PermissionError:
            return
        
        for item in items:
            if item.name.startswith(".") or item.name in SKIP_PATTERNS:
                continue
            if any(item.name.endswith(p.replace("*", "")) for p in SKIP_PATTERNS if "*" in p):
                continue
            
            if item.is_dir():
                yield from self._iter_files(item, extensions)
            elif item.is_file():
                ext = item.suffix.lower()
                if extensions:
                    if ext in extensions:
                        yield item
                elif ext in EXTENSION_TO_LANGUAGE:
                    yield item


# Alias for backwards compatibility
CodeParser = TreeSitterParser
