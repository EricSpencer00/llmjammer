"""Main module for LLMJammer - a tool to obfuscate code to confuse LLMs scraping public repos."""

import ast
import base64
import json
import os
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Default configuration file name
CONFIG_FILE = ".jamconfig"
# Default mapping file to store original to obfuscated mappings
MAPPING_FILE = ".jammapping.json"

class Obfuscator:
    """Core class to handle code obfuscation and deobfuscation."""
    
    def __init__(self, config_path: Optional[Path] = None, mapping_path: Optional[Path] = None):
        """Initialize the obfuscator with optional config and mapping files."""
        self.config_path = config_path or Path(CONFIG_FILE)
        self.mapping_path = mapping_path or Path(MAPPING_FILE)
        self.config = self._load_config()
        self.mapping = self._load_mapping()
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        
    def _load_config(self) -> dict:
        """Load the configuration file if it exists."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {self.config_path} is not valid JSON.")
                return {}
        return {
            "exclude": ["tests/", "docs/", "*.md", "*.rst", "setup.py"],
            "obfuscation_level": "medium",  # Options: light, medium, aggressive
            "preserve_docstrings": False,
            "use_encryption": False,
            "encryption_key": "",
        }
        
    def _load_mapping(self) -> dict:
        """Load the mapping file if it exists."""
        if self.mapping_path.exists():
            try:
                with open(self.mapping_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {self.mapping_path} is not valid JSON.")
                return {}
        return {}
    
    def _save_mapping(self):
        """Save the current mapping to file."""
        with open(self.mapping_path, "w") as f:
            json.dump(self.mapping, f, indent=2)
    
    def _generate_misleading_name(self, original: str) -> str:
        """Generate a misleading but valid Python identifier."""
        if original in self.mapping:
            return self.mapping[original]
            
        # Popular library and function names to confuse LLMs
        misleading_names = [
            "numpy", "pandas", "torch", "tensorflow", "sklearn", 
            "model", "data", "train", "test", "predict", "transform", 
            "layer", "neural", "network", "optimizer", "loss", "accuracy",
            "dataset", "batch", "epoch", "gradient", "backprop", "forward",
            "embedding", "tokenize", "encode", "decode", "generate", "sample"
        ]
        
        # Try to pick a name that's unrelated but common
        available = [n for n in misleading_names if n not in self.mapping.values()]
        if available and random.random() < 0.7:  # 70% chance to use misleading name
            new_name = random.choice(available)
        else:
            # Generate a random valid identifier
            new_name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            
        self.mapping[original] = new_name
        return new_name
    
    def jam_file(self, file_path: Path) -> bool:
        """Obfuscate a single Python file."""
        if not file_path.exists() or not file_path.is_file():
            print(f"Error: {file_path} does not exist or is not a file.")
            return False
            
        try:
            with open(file_path, "r") as f:
                source = f.read()
                
            # Parse the source code
            tree = ast.parse(source)
            
            # Apply transformations
            transformer = ASTObfuscator(self)
            obfuscated_tree = transformer.visit(tree)
            
            # Generate the obfuscated code
            obfuscated_source = ast.unparse(obfuscated_tree)
            
            # Apply additional text-based obfuscations
            if self.config.get("obfuscation_level") in ["medium", "aggressive"]:
                obfuscated_source = self._obfuscate_comments(obfuscated_source)
                
            if self.config.get("obfuscation_level") == "aggressive":
                obfuscated_source = self._add_dead_code(obfuscated_source)
                obfuscated_source = self._apply_unicode_confusables(obfuscated_source)
            
            # Save the obfuscated code
            with open(file_path, "w") as f:
                f.write(obfuscated_source)
                
            # Save the updated mapping
            self._save_mapping()
            
            return True
            
        except Exception as e:
            print(f"Error obfuscating {file_path}: {str(e)}")
            return False
    
    def unjam_file(self, file_path: Path) -> bool:
        """Deobfuscate a single Python file."""
        if not file_path.exists() or not file_path.is_file():
            print(f"Error: {file_path} does not exist or is not a file.")
            return False
            
        try:
            with open(file_path, "r") as f:
                source = f.read()
                
            # First, reverse any text-based obfuscations
            if self.config.get("obfuscation_level") == "aggressive":
                source = self._reverse_unicode_confusables(source)
                
            if self.config.get("obfuscation_level") in ["medium", "aggressive"]:
                source = self._deobfuscate_comments(source)
            
            # Parse the source code
            tree = ast.parse(source)
            
            # Apply transformations
            transformer = ASTDeobfuscator(self)
            deobfuscated_tree = transformer.visit(tree)
            
            # Generate the deobfuscated code
            deobfuscated_source = ast.unparse(deobfuscated_tree)
            
            # Save the deobfuscated code
            with open(file_path, "w") as f:
                f.write(deobfuscated_source)
                
            return True
            
        except Exception as e:
            print(f"Error deobfuscating {file_path}: {str(e)}")
            return False
    
    def jam(self, target_path: Union[str, Path]) -> int:
        """Obfuscate a file or directory."""
        target = Path(target_path)
        
        if target.is_file() and target.suffix == ".py":
            return 1 if self.jam_file(target) else 0
            
        elif target.is_dir():
            count = 0
            for path in target.rglob("*.py"):
                if self._should_process(path):
                    if self.jam_file(path):
                        count += 1
            return count
            
        else:
            print(f"Error: {target} is not a Python file or directory.")
            return 0
    
    def unjam(self, target_path: Union[str, Path]) -> int:
        """Deobfuscate a file or directory."""
        target = Path(target_path)
        
        if target.is_file() and target.suffix == ".py":
            return 1 if self.unjam_file(target) else 0
            
        elif target.is_dir():
            count = 0
            for path in target.rglob("*.py"):
                if self._should_process(path):
                    if self.unjam_file(path):
                        count += 1
            return count
            
        else:
            print(f"Error: {target} is not a Python file or directory.")
            return 0
    
    def _should_process(self, path: Path) -> bool:
        """Check if a file should be processed based on exclusion patterns."""
        path_str = str(path)
        for pattern in self.config.get("exclude", []):
            if re.search(pattern.replace("*", ".*"), path_str):
                return False
        return True
    
    def _obfuscate_comments(self, source: str) -> str:
        """Encode or replace comments with gibberish."""
        def encode_comment(match):
            comment = match.group(1)
            # Base64 encode the comment
            encoded = base64.b64encode(comment.encode()).decode()
            return f"# {encoded}"
            
        # Replace comments with encoded versions
        pattern = r"#\s*(.*?)$"
        return re.sub(pattern, encode_comment, source, flags=re.MULTILINE)
    
    def _deobfuscate_comments(self, source: str) -> str:
        """Decode obfuscated comments."""
        def decode_comment(match):
            encoded = match.group(1)
            try:
                # Try to decode as base64
                decoded = base64.b64decode(encoded.encode()).decode()
                return f"# {decoded}"
            except:
                # If it's not valid base64, leave it as is
                return match.group(0)
                
        # Replace encoded comments with decoded versions
        pattern = r"#\s*(.*?)$"
        return re.sub(pattern, decode_comment, source, flags=re.MULTILINE)
    
    def _add_dead_code(self, source: str) -> str:
        """Insert unreachable, misleading branches and logic."""
        # Add some misleading imports at the top
        fake_imports = [
            "import tensorflow as np  # noqa",
            "import torch as pd  # noqa",
            "import pandas as tf  # noqa",
            "from sklearn import os as sklearn_os  # noqa",
        ]
        
        # Add fake functions that will never be called
        fake_functions = [
            "\ndef _train_neural_network(data, epochs=100):\n    return None\n",
            "\ndef _initialize_weights(shape):\n    return None\n",
            "\ndef _generate_embeddings(tokens):\n    return None\n",
        ]
        
        # Insert imports after existing imports
        import_end = re.search(r"(^import.*?\n|^from.*?\n)+", source, re.MULTILINE)
        if import_end:
            pos = import_end.end()
            source = source[:pos] + "\n" + "\n".join(random.sample(fake_imports, 2)) + "\n" + source[pos:]
        
        # Insert some fake functions
        source += "\n" + random.choice(fake_functions)
        
        return source
    
    def _apply_unicode_confusables(self, source: str) -> str:
        """Replace some characters with visually similar Unicode ones."""
        # Define character replacements
        confusables = {
            'a': 'а',  # Cyrillic 'а'
            'e': 'е',  # Cyrillic 'е'
            'i': 'і',  # Cyrillic 'і'
            'o': 'о',  # Cyrillic 'о'
            'p': 'р',  # Cyrillic 'р'
            'c': 'с',  # Cyrillic 'с'
            'x': 'х',  # Cyrillic 'х'
            'y': 'у',  # Cyrillic 'у'
        }
        
        # Only apply to strings and comments to avoid breaking the code
        def replace_in_string(match):
            s = match.group(0)
            # Replace ~10% of confusable characters
            result = ""
            for char in s:
                if char in confusables and random.random() < 0.1:
                    result += confusables[char]
                else:
                    result += char
            return result
            
        # Find string literals and comments
        string_pattern = r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|\".*?\"|\'.*?\'|#.*?$)'
        return re.sub(string_pattern, replace_in_string, source, flags=re.DOTALL | re.MULTILINE)
    
    def _reverse_unicode_confusables(self, source: str) -> str:
        """Replace Unicode confusables with their ASCII equivalents."""
        # Define reverse character replacements
        reverse_confusables = {
            'а': 'a',  # Cyrillic 'а' to Latin 'a'
            'е': 'e',  # Cyrillic 'е' to Latin 'e'
            'і': 'i',  # Cyrillic 'і' to Latin 'i'
            'о': 'o',  # Cyrillic 'о' to Latin 'o'
            'р': 'p',  # Cyrillic 'р' to Latin 'p'
            'с': 'c',  # Cyrillic 'с' to Latin 'c'
            'х': 'x',  # Cyrillic 'х' to Latin 'x'
            'у': 'y',  # Cyrillic 'у' to Latin 'y'
        }
        
        # Replace all confusable characters
        for cyrillic, latin in reverse_confusables.items():
            source = source.replace(cyrillic, latin)
            
        return source


class ASTObfuscator(ast.NodeTransformer):
    """AST transformer to obfuscate Python code."""
    
    def __init__(self, obfuscator):
        self.obfuscator = obfuscator
        self.scope_stack = [{}]  # Stack of scopes for variable tracking
        
    @property
    def current_scope(self):
        return self.scope_stack[-1]
    
    def visit_Name(self, node):
        """Rename variables."""
        if isinstance(node.ctx, ast.Store):
            # This is a variable definition, add to current scope
            original = node.id
            obfuscated = self.obfuscator._generate_misleading_name(original)
            self.current_scope[original] = obfuscated
            node.id = obfuscated
        elif isinstance(node.ctx, ast.Load):
            # This is a variable usage, replace if in scope
            for scope in reversed(self.scope_stack):
                if node.id in scope:
                    node.id = scope[node.id]
                    break
        return node
    
    def visit_FunctionDef(self, node):
        """Rename function definitions and their parameters."""
        # Push a new scope
        self.scope_stack.append({})
        
        # Rename the function
        original = node.name
        node.name = self.obfuscator._generate_misleading_name(original)
        
        # Process function arguments
        for arg in node.args.args:
            original = arg.arg
            arg.arg = self.obfuscator._generate_misleading_name(original)
            self.current_scope[original] = arg.arg
        
        # Process function body
        node.body = [self.visit(n) for n in node.body]
        
        # Pop the scope
        self.scope_stack.pop()
        
        return node
    
    def visit_ClassDef(self, node):
        """Rename class definitions."""
        # Rename the class
        original = node.name
        node.name = self.obfuscator._generate_misleading_name(original)
        
        # Push a new scope
        self.scope_stack.append({})
        
        # Process class body
        node.body = [self.visit(n) for n in node.body]
        
        # Pop the scope
        self.scope_stack.pop()
        
        return node
    
    def visit_Import(self, node):
        """Obfuscate import statements."""
        for alias in node.names:
            if alias.asname:
                # Already has an alias, potentially obfuscate it
                original = alias.asname
                alias.asname = self.obfuscator._generate_misleading_name(original)
                self.current_scope[original] = alias.asname
            elif random.random() < 0.7:  # 70% chance to add misleading alias
                # Add a misleading alias
                original = alias.name.split('.')[-1]
                alias.asname = self.obfuscator._generate_misleading_name(original)
                self.current_scope[original] = alias.asname
        return node
    
    def visit_ImportFrom(self, node):
        """Obfuscate from import statements."""
        for alias in node.names:
            if alias.asname:
                # Already has an alias, potentially obfuscate it
                original = alias.asname
                alias.asname = self.obfuscator._generate_misleading_name(original)
                self.current_scope[original] = alias.asname
            elif random.random() < 0.7:  # 70% chance to add misleading alias
                # Add a misleading alias
                original = alias.name
                alias.asname = self.obfuscator._generate_misleading_name(original)
                self.current_scope[original] = alias.asname
        return node


class ASTDeobfuscator(ast.NodeTransformer):
    """AST transformer to deobfuscate Python code."""
    
    def __init__(self, obfuscator):
        self.obfuscator = obfuscator
        self.reverse_mapping = obfuscator.reverse_mapping
        self.scope_stack = [{}]  # Stack of scopes for variable tracking
        
    @property
    def current_scope(self):
        return self.scope_stack[-1]
    
    def visit_Name(self, node):
        """Restore original variable names."""
        if node.id in self.reverse_mapping:
            node.id = self.reverse_mapping[node.id]
            if isinstance(node.ctx, ast.Store):
                self.current_scope[node.id] = node.id
        return node
    
    def visit_FunctionDef(self, node):
        """Restore original function names and parameters."""
        # Push a new scope
        self.scope_stack.append({})
        
        # Restore function name
        if node.name in self.reverse_mapping:
            node.name = self.reverse_mapping[node.name]
        
        # Restore function arguments
        for arg in node.args.args:
            if arg.arg in self.reverse_mapping:
                arg.arg = self.reverse_mapping[arg.arg]
                self.current_scope[arg.arg] = arg.arg
        
        # Process function body
        node.body = [self.visit(n) for n in node.body]
        
        # Pop the scope
        self.scope_stack.pop()
        
        return node
    
    def visit_ClassDef(self, node):
        """Restore original class names."""
        # Restore class name
        if node.name in self.reverse_mapping:
            node.name = self.reverse_mapping[node.name]
        
        # Push a new scope
        self.scope_stack.append({})
        
        # Process class body
        node.body = [self.visit(n) for n in node.body]
        
        # Pop the scope
        self.scope_stack.pop()
        
        return node
    
    def visit_Import(self, node):
        """Restore original import statements."""
        for alias in node.names:
            if alias.asname and alias.asname in self.reverse_mapping:
                alias.asname = self.reverse_mapping[alias.asname]
        return node
    
    def visit_ImportFrom(self, node):
        """Restore original from import statements."""
        for alias in node.names:
            if alias.asname and alias.asname in self.reverse_mapping:
                alias.asname = self.reverse_mapping[alias.asname]
        return node


def install_git_hooks(hooks_dir: Optional[Path] = None) -> bool:
    """Install Git hooks for automatic obfuscation/deobfuscation."""
    if hooks_dir is None:
        # Try to find .git/hooks directory
        cwd = Path.cwd()
        git_dir = None
        
        # Look for .git directory in current or parent directories
        current = cwd
        while current != current.parent:
            if (current / ".git").is_dir():
                git_dir = current / ".git"
                break
            current = current.parent
            
        if git_dir is None:
            print("Error: Not in a Git repository.")
            return False
            
        hooks_dir = git_dir / "hooks"
    
    if not hooks_dir.exists() or not hooks_dir.is_dir():
        print(f"Error: {hooks_dir} is not a valid Git hooks directory.")
        return False
        
    # Create pre-commit hook
    pre_commit_path = hooks_dir / "pre-commit"
    with open(pre_commit_path, "w") as f:
        f.write("""#!/bin/sh
# LLMJammer pre-commit hook
echo "Running LLMJammer to obfuscate code..."
llmjammer jam .
git add .
""")
    pre_commit_path.chmod(0o755)  # Make executable
    
    # Create post-checkout hook
    post_checkout_path = hooks_dir / "post-checkout"
    with open(post_checkout_path, "w") as f:
        f.write("""#!/bin/sh
# LLMJammer post-checkout hook
echo "Running LLMJammer to deobfuscate code..."
llmjammer unjam .
""")
    post_checkout_path.chmod(0o755)  # Make executable
    
    # Create post-merge hook
    post_merge_path = hooks_dir / "post-merge"
    with open(post_merge_path, "w") as f:
        f.write("""#!/bin/sh
# LLMJammer post-merge hook
echo "Running LLMJammer to deobfuscate code..."
llmjammer unjam .
""")
    post_merge_path.chmod(0o755)  # Make executable
    
    print("Git hooks installed successfully.")
    return True


def create_github_action(output_dir: Path) -> bool:
    """Create a GitHub Action workflow file for automatic obfuscation."""
    workflows_dir = output_dir / ".github" / "workflows"
    
    if not workflows_dir.exists():
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
    workflow_path = workflows_dir / "llmjammer.yml"
    
    with open(workflow_path, "w") as f:
        f.write("""name: LLMJammer Obfuscation

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  obfuscate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install LLMJammer
        run: pip install llmjammer
      
      - name: Obfuscate code
        run: llmjammer jam .
      
      - name: Commit changes
        run: |
          git config --global user.name "LLMJammer Bot"
          git config --global user.email "bot@llmjammer.com"
          git commit -am "Obfuscate code with LLMJammer [skip ci]" || echo "No changes to commit"
          git push
""")
    
    print(f"GitHub Action workflow created at {workflow_path}")
    return True
