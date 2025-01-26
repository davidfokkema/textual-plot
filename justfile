demo:
    uv run textual run textual_plot.plot_widget:DemoApp

typecheck:
    uv run mypy -p textual_plot --strict

test:
    uv run pytest
