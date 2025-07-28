#!/usr/bin/env python

"""Tests for `llmjammer` package."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from llmjammer import Obfuscator


@pytest.fixture
def sample_python_file():
    """Create a temporary Python file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        content = """
def calculate_model_accuracy(model, test_data, labels):
    \"\"\"
    Calculate the accuracy of a machine learning model.
    
    Args:
        model: The trained model
        test_data: Test dataset
        labels: Ground truth labels
        
    Returns:
        float: The accuracy score
    \"\"\"
    predictions = model.predict(test_data)
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(labels)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            # Training logic would go here
            pass
            
    def predict(self, data):
        # Prediction logic would go here
        return [0] * len(data)
"""
        tmp.write(content.encode())
        tmp_path = Path(tmp.name)
    
    yield tmp_path
    
    # Clean up
    if tmp_path.exists():
        os.unlink(tmp_path)


@pytest.fixture
def config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        config = {
            "exclude": ["tests/", "docs/"],
            "obfuscation_level": "medium",
            "preserve_docstrings": False,
            "use_encryption": False,
            "encryption_key": "",
        }
        tmp.write(json.dumps(config).encode())
        tmp_path = Path(tmp.name)
    
    yield tmp_path
    
    # Clean up
    if tmp_path.exists():
        os.unlink(tmp_path)


def test_obfuscator_initialization():
    """Test that the Obfuscator initializes properly."""
    # Create a temporary mapping file to avoid persistence issues
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(b"{}")  # Empty mapping
        mapping_path = Path(tmp.name)
    
    try:
        obfuscator = Obfuscator(mapping_path=mapping_path)
        assert obfuscator is not None
        assert obfuscator.config is not None
        assert obfuscator.mapping == {}
    finally:
        # Clean up
        if mapping_path.exists():
            os.unlink(mapping_path)


def test_jam_and_unjam(sample_python_file, config_file):
    """Test the obfuscation and deobfuscation process."""
    # Read the original content
    with open(sample_python_file, "r") as f:
        original_content = f.read()
    
    # Create an obfuscator with our test config
    obfuscator = Obfuscator(config_file)
    
    # Obfuscate the file
    success = obfuscator.jam_file(sample_python_file)
    assert success, "Obfuscation should succeed"
    
    # Read the obfuscated content
    with open(sample_python_file, "r") as f:
        obfuscated_content = f.read()
    
    # Verify that the content changed
    assert original_content != obfuscated_content, "Content should be different after obfuscation"
    
    # Check that a mapping was created
    assert len(obfuscator.mapping) > 0, "A mapping should be created"
    
    # Deobfuscate the file
    success = obfuscator.unjam_file(sample_python_file)
    assert success, "Deobfuscation should succeed"
    
    # Read the deobfuscated content
    with open(sample_python_file, "r") as f:
        deobfuscated_content = f.read()
    
    # Compare with original (may have formatting differences due to AST operations)
    assert "calculate_model_accuracy" in deobfuscated_content, "Original function name should be restored"
    assert "NeuralNetwork" in deobfuscated_content, "Original class name should be restored"
    

def test_excluded_files(sample_python_file):
    """Test that excluded files are not obfuscated."""
    # Create a custom config that excludes our test file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        filename = os.path.basename(sample_python_file)
        config = {
            "exclude": [filename],
            "obfuscation_level": "medium",
        }
        tmp.write(json.dumps(config).encode())
        config_path = Path(tmp.name)
    
    try:
        # Create an obfuscator with our custom config
        obfuscator = Obfuscator(config_path)
        
        # Try to obfuscate the file (should be skipped)
        count = obfuscator.jam(sample_python_file.parent)
        
        # Since our file should be excluded, no files should be obfuscated
        assert count == 0, "No files should be obfuscated when excluded"
    finally:
        # Clean up
        if config_path.exists():
            os.unlink(config_path)


def test_self_protection():
    """Test that llmjammer doesn't obfuscate itself."""
    obfuscator = Obfuscator()
    
    # Test that llmjammer files are excluded
    llmjammer_files = [
        Path("src/llmjammer/llmjammer.py"),
        Path("src/llmjammer/cli.py"),
        Path("src/llmjammer/__init__.py"),
    ]
    
    for file_path in llmjammer_files:
        if file_path.exists():
            should_process = obfuscator._should_process(file_path)
            assert not should_process, f"{file_path} should not be processed"


def test_comment_obfuscation_safety():
    """Test that comment obfuscation doesn't create syntax errors."""
    obfuscator = Obfuscator()
    
    # Test with comments that could cause issues
    test_cases = [
        '''# This is a test comment with "quotes" and {braces}
def test_function():
    # Another comment with 'single quotes'
    return "hello"
''',
        '''# Comment with backslash \\
def test():
    pass
''',
        '''# Comment with newline
def test():
    pass
''',
    ]
    
    for i, test_source in enumerate(test_cases):
        # Try to obfuscate comments
        obfuscated = obfuscator._obfuscate_comments(test_source)
        
        # Verify the result is valid Python
        try:
            compile(obfuscated, f'<test_case_{i}>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Comment obfuscation created syntax error in test case {i}: {e}")
