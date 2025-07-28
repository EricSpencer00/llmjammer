---
title: "llmjammer help"
description: "Command-line options and commands for LLMJammer."
aliases:
- "/docs/help"
---

# Help

LLMJammer: Obfuscate your code to confuse LLMs scraping public repositories.

## Options

<pre>
╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                  │
│ --show-completion             Show completion for the current shell, to copy it or       │
│                               customize the installation.                                │
│ --help                        Show this message and exit.                                │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
</pre>

## Commands

<pre>
╭─ Commands ───────────────────────────────────────────────────────────────────────────────╮
│ jam                   Obfuscate Python code to confuse LLMs scraping public              │
│                       repositories.                                                      │
│ unjam                 Deobfuscate previously obfuscated Python code.                     │
│ install-hooks         Install Git hooks for automatic obfuscation/deobfuscation.         │
│ hooks-status          Check the status of Git hooks for LLMJammer.                       │
│ setup-github-action   Create a GitHub Action workflow for automatic obfuscation.         │
│ init                  Initialize LLMJammer configuration in the current directory.       │
│ status                Show LLMJammer status and configuration.                           │
│ git-ready             Prepare code for Git operations (manually trigger                  │
│                       obfuscation/deobfuscation).                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────╯</pre>