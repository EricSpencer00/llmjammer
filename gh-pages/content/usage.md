---
title: "Usage"
description: "Learn how to use LLMJammer."
---

# Usage

## Command Line Interface

LLMJammer provides a simple command-line interface for obfuscating and deobfuscating your Python code.

### Initialize a Project

```bash
cd your-project
llmjammer init
```

This will create a `.jamconfig` file with default settings and offer to set up Git hooks and GitHub Actions.

### Obfuscate Code (Jam)

```bash
# Obfuscate all Python files in current directory
llmjammer jam .

# Obfuscate a specific file
llmjammer jam path/to/file.py

# Use a custom configuration file
llmjammer jam . --config path/to/config.json
```

### Deobfuscate Code (Unjam)

```bash
# Deobfuscate all Python files in current directory
llmjammer unjam .

# Deobfuscate a specific file
llmjammer unjam path/to/file.py

# Use a custom mapping file
llmjammer unjam . --mapping path/to/mapping.json
```

### Install Git Hooks

```bash
# Install Git hooks in the current repository
llmjammer install-hooks

# Install Git hooks in a specific repository
llmjammer install-hooks --hooks-dir /path/to/repo/.git/hooks
```

### Set Up GitHub Actions

```bash
# Create a GitHub Action workflow for automatic obfuscation
llmjammer setup-github-action
```

### Check Status

```bash
# View LLMJammer status and configuration
llmjammer status
```

## Python API

You can also use LLMJammer programmatically in your Python code:

```python
from llmjammer import Obfuscator

# Create an obfuscator
obfuscator = Obfuscator()

# Obfuscate a file or directory
obfuscator.jam("path/to/file.py")  # Single file
obfuscator.jam("path/to/directory")  # Directory (recursive)

# Deobfuscate
obfuscator.unjam("path/to/file.py")
obfuscator.unjam("path/to/directory")

# With custom config and mapping files
obfuscator = Obfuscator(
    config_path="path/to/config.json", 
    mapping_path="path/to/mapping.json"
)
```
