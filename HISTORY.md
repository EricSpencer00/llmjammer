# History

## 0.0.2 (2025-01-27)

* Fixed critical bug where llmjammer could corrupt its own installation
* Improved comment obfuscation to prevent syntax errors from problematic characters
* Added self-protection mechanism to exclude llmjammer package from obfuscation
* Enhanced validation in comment encoding/decoding to handle quotes, braces, and backslashes
* Replaced f-strings with safer string concatenation in comment obfuscation
* Added comprehensive tests for self-protection and comment obfuscation safety
* Improved error handling and validation throughout the obfuscation process

## 0.0.1 (2025-07-26)

* First release on PyPI.
