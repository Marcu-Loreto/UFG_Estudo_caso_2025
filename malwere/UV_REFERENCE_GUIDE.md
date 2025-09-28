# UV Package Manager - Complete Reference Guide

## What is UV?
UV is a fast Python package installer and resolver, written in Rust. It's designed to be a drop-in replacement for pip and pip-tools, but much faster.

## Installation
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## Project Management

### Create New Project
```bash
# Create new project
uv init my-project
cd my-project

# Create with specific Python version
uv init --python 3.12 my-project
```

### Initialize Existing Project
```bash
# Initialize current directory as UV project
uv init --no-readme
```

## Dependency Management

### Add Dependencies
```bash
# Add single package
uv add numpy

# Add multiple packages
uv add pandas scikit-learn plotly

# Add with version constraints
uv add "pandas>=2.0.0"
uv add "numpy>=1.24.0,<2.0.0"

# Add development dependencies
uv add --dev pytest black flake8 mypy

# Add optional dependencies
uv add --optional web fastapi uvicorn
```

### Remove Dependencies
```bash
# Remove single package
uv remove numpy

# Remove multiple packages
uv remove pandas scikit-learn

# Remove development dependencies
uv remove --dev pytest
```

### Update Dependencies
```bash
# Update all packages to latest compatible versions
uv sync --upgrade

# Update specific package
uv add --upgrade pandas

# Update to latest versions (may break compatibility)
uv sync --upgrade-package pandas
```

## Installation and Sync

### Install Dependencies
```bash
# Install all dependencies from pyproject.toml
uv sync

# Install without dev dependencies
uv sync --no-dev

# Install with specific groups
uv sync --group dev

# Install and lock dependencies
uv lock
uv sync
```

### Run Commands
```bash
# Run Python script
uv run python script.py

# Run with specific dependencies
uv run --with requests python script.py

# Run in virtual environment
uv run --python 3.12 python script.py
```

## Virtual Environment Management

### Create Virtual Environment
```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.12

# Create in specific directory
uv venv .venv
```

### Activate Virtual Environment
```bash
# On Windows
.venv\Scripts\activate

# On Unix/macOS
source .venv/bin/activate
```

## Project Configuration (pyproject.toml)

### Basic Structure
```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My awesome project"
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]
web = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "black>=23.0.0",
]
```

## Common Commands Summary

### Project Setup
```bash
uv init                    # Initialize new project
uv init --no-readme        # Initialize without README
uv sync                    # Install dependencies
uv run python script.py    # Run script with dependencies
```

### Dependency Management
```bash
uv add package             # Add dependency
uv add --dev package       # Add dev dependency
uv remove package          # Remove dependency
uv sync --upgrade          # Update all dependencies
uv lock                    # Lock dependency versions
```

### Environment Management
```bash
uv venv                    # Create virtual environment
uv run --python 3.12       # Run with specific Python version
uv run --with requests     # Run with additional packages
```

## Migration from pip/requirements.txt

### From requirements.txt
```bash
# Create pyproject.toml from requirements.txt
uv init
# Then manually add dependencies from requirements.txt using:
uv add $(cat requirements.txt)
```

### From pip freeze
```bash
# Generate requirements.txt from current environment
pip freeze > requirements.txt

# Then convert to UV project
uv init
# Add dependencies one by one or edit pyproject.toml manually
```

## Performance Benefits

- **10-100x faster** than pip
- **Parallel downloads** and installations
- **Better dependency resolution**
- **Rust-based** for maximum performance
- **Compatible** with existing pip workflows

## Best Practices

1. **Use pyproject.toml** instead of requirements.txt
2. **Pin major versions** but allow minor updates: `"pandas>=2.0.0,<3.0.0"`
3. **Separate dev dependencies** from production dependencies
4. **Use uv lock** for reproducible builds
5. **Use uv run** for scripts instead of activating venv manually

## Troubleshooting

### Common Issues
```bash
# Clear cache if issues occur
uv cache clean

# Reinstall everything
rm -rf .venv
uv sync

# Check for conflicts
uv tree

# Show dependency graph
uv show --tree
```

### Debug Commands
```bash
uv show                    # Show installed packages
uv show package            # Show specific package info
uv tree                    # Show dependency tree
uv list                    # List all installed packages
```

## Integration with IDEs

### VS Code
- Install "Python" extension
- Select UV virtual environment: `Ctrl+Shift+P` → "Python: Select Interpreter"
- Choose `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Unix)

### PyCharm
- File → Settings → Project → Python Interpreter
- Add Interpreter → Existing Environment
- Select `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Unix)

## Example Workflow

```bash
# 1. Create new project
uv init my-ml-project
cd my-ml-project

# 2. Add dependencies
uv add pandas numpy scikit-learn plotly

# 3. Add dev dependencies
uv add --dev pytest black jupyter

# 4. Create and run script
echo "import pandas as pd; print('Hello UV!')" > main.py
uv run python main.py

# 5. Update dependencies
uv sync --upgrade
```

## UV vs Other Tools

| Feature | UV | pip | Poetry | pipenv |
|---------|----|----|---------|--------|
| Speed | ⚡⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡ |
| Lock file | ✅ | ❌ | ✅ | ✅ |
| pyproject.toml | ✅ | ❌ | ✅ | ❌ |
| Virtual env | ✅ | ❌ | ✅ | ✅ |
| Dependency groups | ✅ | ❌ | ✅ | ✅ |
| Rust-based | ✅ | ❌ | ❌ | ❌ |

---

**Created:** $(date)  
**UV Version:** Check with `uv --version`  
**Python Version:** Check with `uv run python --version`
