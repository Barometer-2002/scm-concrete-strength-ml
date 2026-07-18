# Contributing

1. Create a focused branch from `main`.
2. Install the development environment with `python -m pip install -e ".[dev]"`.
3. Run `ruff check .` and `pytest` before opening a pull request.
4. Add tests for behavior changes and keep public examples independent of the
   private research dataset.

Bug reports should include a minimal input schema, Python version, package
versions, the command used, and the full traceback. Do not attach proprietary
mixture records or credentials to an issue.
