# Installation

## Stable Release

To install LLMJammer, run this command in your terminal:

```sh
pip install llmjammer
```

This is the preferred method to install LLMJammer, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Requirements

LLMJammer requires:

- Python 3.10 or higher
- Git (for Git hook integration)

## From Sources

The sources for LLMJammer can be downloaded from the [Github repo](https://github.com/EricSpencer00/llmjammer).

You can either clone the public repository:

```sh
git clone git://github.com/EricSpencer00/llmjammer
```

Or download the [tarball](https://github.com/EricSpencer00/llmjammer/tarball/master):

```sh
curl -OJL https://github.com/EricSpencer00/llmjammer/tarball/master
```

Once you have a copy of the source, you can install it with:

```sh
pip install -e .
```

For development installation with testing dependencies:

```sh
pip install -e ".[test]"
```

## Quick Setup

After installation, you can quickly set up LLMJammer in your project:

```sh
cd your-project
llmjammer init
```

This will:

1. Create a default configuration file (`.jamconfig`)
2. Offer to install Git hooks for automatic obfuscation/deobfuscation
3. Offer to set up a GitHub Action for CI/CD integration

## Verifying Installation

To verify that LLMJammer was installed correctly:

```sh
llmjammer --help
```

You should see the help message with all available commands.
