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


@app.command("validate-data")
def validate_data() -> None:
    """Load and validate the fact sheet and question set, then print counts."""
    from faq_eval.dataset import load_fact_sheet, load_questions, summarise

    sheet = load_fact_sheet()
    questions = load_questions()
    answerable = sum(q.answerable for q in questions)
    typer.echo(f"fact sheet: {len(sheet.split())} words")
    typer.echo(f"questions: {len(questions)} ({answerable} answerable, {len(questions) - answerable} unanswerable)")
    for category, n in summarise(questions).items():
        typer.echo(f"  {category:<14} {n}")


if __name__ == "__main__":
    app()
