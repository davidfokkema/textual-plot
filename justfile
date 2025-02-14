demo:
    uv run textual run textual_plot.demo:DemoApp

typecheck:
    uv run mypy -p textual_plot --strict

test:
    uv run pytest

format:
    uvx ruff format

fix:
    uvx ruff check --fix
