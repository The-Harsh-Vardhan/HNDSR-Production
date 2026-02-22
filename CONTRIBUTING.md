# Contributing to HNDSR-Production

First off, thank you for considering contributing to HNDSR-Production! It's people like you that make high-quality ML engineering tools possible.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs
- Use the [**Bug Report**](.github/ISSUE_TEMPLATE/bug_report.yml) template.
- Describe the exact steps to reproduce the issue.
- Include environment details (Python version, Torch version, GPU model, OS).

### Suggesting Enhancements
- Use the [**Feature Request**](.github/ISSUE_TEMPLATE/feature_request.yml) template.
- Explain why this enhancement would be useful to the broader community.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. Install dev dependencies: `pip install -r requirements.txt`.
3. If you've added code that should be tested, add tests in the `tests/` directory.
4. Ensure the test suite passes: `pytest tests/ -v`.
5. Run formatting and linting: `black .` and `flake8 .`.
6. Submit a PR with a clear description of your changes.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/HNDSR-Production.git
cd HNDSR-Production

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install all dependencies (dev + test + prod)
pip install -r requirements.txt

# Run the linter and tests
black --check .
flake8 .
pytest tests/ -v
```

## Coding Standards
- We follow **PEP 8** (enforced by `black` and `flake8`).
- Use type hints for all function signatures.
- Write docstrings for all public classes and methods.
- Keep the `README.md` and `architecture.md` updated with any structural changes.

## Commit Messages

Use clear, descriptive commit messages:
- `fix: resolve EMA device mismatch in training loop`
- `feat: add DDIM step parameter to /infer endpoint`
- `docs: update API reference in README`
- `test: add shape validation tests for FNO output`

## Contact

For major architectural changes, please open an issue first to discuss your ideas.
