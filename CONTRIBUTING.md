# Contributing to HNDSR-Production

First off, thank you for considering contributing to HNDSR-Production! It's people like you that make high-quality ML engineering tools possible.

## Code of Conduct
By participating in this project, you are expected to uphold our Code of Conduct (be kind, professional, and respectful).

## How Can I Contribute?

### Reporting Bugs
- Use the **Bug Report** template.
- Describe the exact steps to reproduce the issue.
- Include environment details (Python version, Torch version, GPU model).

### Suggesting Enhancements
- Use the **Feature Request** template.
- Explain why this enhancement would be useful to the broader community.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. Install dev dependencies: `pip install -r requirements.txt`.
3. If you've added code that should be tested, add tests in the `tests/` directory.
4. Ensure the test suite passes: `pytest`.
5. Run linting: `black .` and `flake8 .`.
6. Submit a PR with a clear description of the transition.

## Coding Standards
- We follow **PEP 8**.
- Use type hints for all function signatures.
- Write docstrings for all public classes and methods.
- Keep the `README.md` and `architecture.md` updated with any structural changes.

## Contact
For major architectural changes, please open an issue first to discuss your ideas.
