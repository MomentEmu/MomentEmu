# MomentEmu Documentation

This directory contains the source files for the MomentEmu documentation website, built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Building Documentation Locally

### Prerequisites

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
```

Or install with the docs dependencies:

```bash
pip install -e ".[docs]"
```

### Serve Documentation

```bash
mkdocs serve
```

Then visit http://localhost:8000 to view the documentation.

### Build Documentation

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

## Documentation Structure

- `docs/index.md` - Homepage
- `docs/installation/` - Installation guides
- `docs/tutorials/` - Step-by-step tutorials
- `docs/api/` - API reference documentation
- `docs/theory/` - Mathematical background
- `docs/examples/` - Use cases and applications
- `docs/contributing/` - Development guides

## Contributing to Documentation

1. Edit the Markdown files in the `docs/` directory
2. Test locally with `mkdocs serve`
3. Submit a pull request

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.

## Configuration

- `mkdocs.yml` - Main configuration file
- `docs/javascripts/mathjax.js` - Mathematical notation support
- `.github/workflows/docs.yml` - Automated deployment
