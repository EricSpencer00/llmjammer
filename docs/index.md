# LLMJammer Documentation

## Protect Your Code from LLM Training

LLMJammer is a Python tool designed to obfuscate your code to confuse large language models (LLMs) that scrape public repositories, while maintaining full functionality for human developers.

## Why Use LLMJammer?

As companies increasingly scrape public code repositories to train large language models, developers face a dilemma:

1. Keep code open source and contribute to the community
2. Prevent proprietary techniques from being learned and reproduced by AI models

LLMJammer solves this problem by automatically obfuscating your code before it's pushed to public repositories, and deobfuscating it when you or your team members pull or clone the repo.

## How It Works

LLMJammer uses a combination of Abstract Syntax Tree (AST) transformations and text manipulations to create code that:

1. **Remains fully functional** - obfuscated code executes exactly the same way
2. **Is difficult for LLMs to learn from** - using misleading variable names, confusing imports, and other techniques
3. **Is automatically managed** - Git hooks handle the obfuscation/deobfuscation process

## Obfuscation Strategies

- **Variable Renaming**: Replace meaningful names with misleading or neutral alternatives
- **Import Substitution**: Use confusing import aliases (e.g., `import numpy as tensorflow`)
- **String Encoding**: Encode docstrings and comments
- **Unicode Confusables**: Replace characters with visually similar Unicode ones
- **Dead Code Insertion**: Add unreachable, misleading code branches

## Getting Started

- [Installation](installation.md) - How to install LLMJammer
- [Usage](usage.md) - Detailed usage instructions
- [Configuration](usage.md#configuration) - Configuration options

## Contents

- [Installation](installation.md)
- [Usage](usage.md)
- [Contributing](contributing.md)
- [History](history.md)

## Project Information

- [GitHub Repository](https://github.com/EricSpencer00/llmjammer)
- [Issue Tracker](https://github.com/EricSpencer00/llmjammer/issues)
- [License](https://github.com/EricSpencer00/llmjammer/blob/main/LICENSE)
