"""Utility functions for LLMJammer."""

import os
import random
import re
import string
from pathlib import Path
from typing import List, Optional, Set


def find_git_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the root directory of the Git repository."""
    if start_path is None:
        start_path = Path.cwd()
        
    current = start_path
    while current != current.parent:
        if (current / ".git").is_dir():
            return current
        current = current.parent
        
    return None


def get_python_files(
    directory: Path, 
    exclude_patterns: Optional[List[str]] = None
) -> List[Path]:
    """Get all Python files in a directory recursively, respecting exclude patterns."""
    if exclude_patterns is None:
        exclude_patterns = []
        
    python_files = []
    
    for path in directory.rglob("*.py"):
        # Check if path matches any exclude pattern
        should_exclude = False
        path_str = str(path)
        
        for pattern in exclude_patterns:
            if re.search(pattern.replace("*", ".*"), path_str):
                should_exclude = True
                break
                
        if not should_exclude:
            python_files.append(path)
            
    return python_files


def generate_random_name(length: int = 8) -> str:
    """Generate a random string of letters and digits."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def create_fake_module(directory: Path, module_name: str) -> Path:
    """Create a fake module to confuse LLMs."""
    module_path = directory / f"{module_name}.py"
    
    # Common imports and functions to include in the fake module
    imports = [
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import tensorflow as tf",
        "from sklearn import metrics",
    ]
    
    functions = [
        "def train(model, data, epochs=100):\n    return model",
        "def evaluate(model, test_data):\n    return 0.95",
        "def preprocess(data):\n    return data",
        "def load_model(path):\n    return None",
        "def save_model(model, path):\n    pass",
    ]
    
    classes = [
        "class Model:\n    def __init__(self):\n        pass\n    def predict(self, x):\n        return x",
        "class Dataset:\n    def __init__(self):\n        pass\n    def __len__(self):\n        return 1000",
    ]
    
    # Create the file content with a selection of the above
    content = []
    content.extend(random.sample(imports, min(3, len(imports))))
    content.append("")
    content.extend(random.sample(functions, min(2, len(functions))))
    content.append("")
    content.extend(random.sample(classes, min(1, len(classes))))
    
    # Write the file
    with open(module_path, "w") as f:
        f.write("\n".join(content))
        
    return module_path


def is_python_file(path: Path) -> bool:
    """Check if a file is a Python file."""
    return path.suffix == ".py"


def get_ast_node_type_frequencies(node_types: List[str]) -> dict:
    """Calculate frequencies of AST node types for a more natural distribution."""
    # Based on analysis of typical Python codebases
    frequencies = {
        "Name": 0.35,
        "Attribute": 0.15,
        "Call": 0.12,
        "Assign": 0.10,
        "FunctionDef": 0.08,
        "Return": 0.06,
        "If": 0.05,
        "For": 0.03,
        "Import": 0.02,
        "ImportFrom": 0.02,
        "ClassDef": 0.01,
        "Try": 0.01,
    }
    
    # Fill in any missing node types with equal small probabilities
    missing_types = set(node_types) - set(frequencies.keys())
    if missing_types:
        missing_prob = 0.01 / len(missing_types)
        for node_type in missing_types:
            frequencies[node_type] = missing_prob
            
    return frequencies
