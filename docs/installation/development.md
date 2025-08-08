# Development Installation

Set up MomentEmu for development and contribution.

## Prerequisites

- Python 3.7 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zzhang0123/MomentEmu.git
cd MomentEmu
```

### 2. Create Virtual Environment

=== "venv"
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

=== "conda"
    ```bash
    conda create -n momentemu python=3.10
    conda activate momentemu
    ```

### 3. Install in Development Mode

```bash
# Install with all development dependencies
pip install -e ".[all]"

# Additional development tools
pip install pytest pytest-cov black isort flake8 mypy
```

## Development Dependencies

The `[all]` installation includes:

- **Core dependencies**: numpy, scipy, sympy, scikit-learn
- **Auto-diff frameworks**: jax, jaxlib, torch
- **Visualization**: matplotlib
- **Testing**: pytest, pytest-cov (install separately)
- **Code quality**: black, isort, flake8, mypy (install separately)

## Project Structure

```
MomentEmu/
├── src/MomentEmu/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── MomentEmu.py        # Core emulator class
│   ├── jax_momentemu.py    # JAX integration
│   ├── torch_momentemu.py  # PyTorch integration
│   └── symbolic_momentemu.py # SymPy integration
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_autodiff_variants.py
├── docs/                    # Documentation
├── .github/workflows/       # CI/CD
├── pyproject.toml          # Package configuration
├── mkdocs.yml              # Documentation config
└── README.md               # Main documentation
```

## Running Tests

### Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/MomentEmu --cov-report=html

# Run specific test file
pytest tests/test_autodiff_variants.py -v
```

### Individual Module Tests

```bash
# Test core functionality
python src/MomentEmu/MomentEmu.py

# Test auto-diff modules (if dependencies installed)
python src/MomentEmu/jax_momentemu.py
python src/MomentEmu/torch_momentemu.py
python src/MomentEmu/symbolic_momentemu.py
```

## Code Quality

### Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting

```bash
# Check code style
flake8 src/ tests/

# Type checking
mypy src/MomentEmu/
```

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Building Documentation

### Local Development

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

The documentation will be available at `http://localhost:8000`.

### Documentation Structure

- **docs/**: Source files in Markdown
- **mkdocs.yml**: Configuration file
- **site/**: Generated HTML (git-ignored)

## Contributing Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following existing style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Quality Checks

```bash
# Run full test suite
pytest tests/ -v

# Check code quality
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/

# Type checking
mypy src/MomentEmu/
```

### 4. Submit Pull Request

```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Release Process

### Version Bumping

Update version in `pyproject.toml`:

```toml
[project]
version = "1.1.0"  # Update this
```

### Creating Releases

```bash
# Tag the release
git tag v1.1.0
git push origin v1.1.0

# GitHub Actions will automatically:
# - Run tests
# - Build documentation
# - Create release artifacts
```

## Troubleshooting Development Setup

### Common Issues

!!! warning "Import Errors"
    - Make sure you installed in development mode: `pip install -e .`
    - Check that your virtual environment is activated
    - Verify Python path includes the project directory

!!! warning "Test Failures"
    - Ensure all dependencies are installed: `pip install -e ".[all]"`
    - Check Python version compatibility (3.7+)
    - Run tests in isolation: `pytest tests/test_autodiff_variants.py::test_name -v`

!!! warning "Documentation Build Errors"
    - Install docs dependencies: `pip install mkdocs mkdocs-material mkdocstrings[python]`
    - Check for syntax errors in markdown files
    - Verify `mkdocs.yml` configuration

### Getting Help

- Check existing [GitHub Issues](https://github.com/zzhang0123/MomentEmu/issues)
- Create new issue for bugs or feature requests
- Join discussions in [GitHub Discussions](https://github.com/zzhang0123/MomentEmu/discussions)

## Next Steps

- Read [Contributing Guidelines](development.md)
- Explore [Testing Framework](testing.md)
- Review [Release Process](release-process.md)
