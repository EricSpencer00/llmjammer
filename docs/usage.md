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

## Git Integration

Once Git hooks are installed, LLMJammer works automatically:

1. When you `git commit`, your code is obfuscated before being committed
2. When you `git pull` or `git checkout`, your code is deobfuscated after the operation

This ensures that your repository always contains obfuscated code, but you're always working with the readable version.

## Configuration

LLMJammer can be configured through a `.jamconfig` file:

```json
{
  "exclude": ["tests/", "docs/", "*.md", "*.rst", "setup.py"],
  "obfuscation_level": "medium",
  "preserve_docstrings": false,
  "use_encryption": false,
  "encryption_key": ""
}
```

### Configuration Options

- **exclude**: List of glob patterns for files/directories to skip
- **obfuscation_level**: 
  - "light": Basic variable renaming and import obfuscation
  - "medium": Adds comment encoding and more aggressive renaming
  - "aggressive": Adds Unicode confusables and dead code insertion
- **preserve_docstrings**: When true, docstrings remain readable
- **use_encryption**: Enable additional encryption (future feature)
- **encryption_key**: Secret key for encryption (future feature)
