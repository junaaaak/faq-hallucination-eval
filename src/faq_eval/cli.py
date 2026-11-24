"""Command-line entry point. Commands are added as each phase lands."""

import typer

app = typer.Typer(help="College FAQ bot hallucination eval.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """College FAQ bot hallucination eval."""


@app.command()
def version() -> None:
    """Print the package version."""
    from faq_eval import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
