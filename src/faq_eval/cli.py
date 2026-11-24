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


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the FAQ bot."),
    model: str = typer.Option("llama3.2:3b", help="Ollama model name."),
    prompt: str = typer.Option("strict", help="Prompt variant: lenient or strict."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
) -> None:
    """Ask the FAQ bot a single question."""
    from faq_eval.bot import BotConfig, FAQBot

    bot = FAQBot(BotConfig(model=model, prompt=prompt, temperature=temperature))
    resp = bot.ask(question)
    typer.echo(resp.answer)
    typer.echo(f"\n[{resp.latency_s:.2f}s, {resp.prompt_tokens} prompt + {resp.completion_tokens} completion tokens]")


SMOKE_IDS = ["a001", "a048", "u013", "u037", "u049", "u001"]


@app.command()
def smoke(
    model: str = typer.Option("llama3.2:3b", help="Ollama model name."),
    prompt: str = typer.Option("strict", help="Prompt variant: lenient or strict."),
) -> None:
    """Run a handful of questions (one per kind) and print the raw answers."""
    from faq_eval.bot import BotConfig, FAQBot
    from faq_eval.dataset import load_questions

    by_id = {q.id: q for q in load_questions()}
    bot = FAQBot(BotConfig(model=model, prompt=prompt))
    for qid in SMOKE_IDS:
        q = by_id[qid]
        resp = bot.ask(q.question)
        expected = q.required_facts or q.premise_corrections or "refusal"
        typer.echo(f"[{q.id} | {q.category.value} | {resp.latency_s:.1f}s] {q.question}")
        typer.echo(f"  expected: {expected}")
        typer.echo(f"  answer:   {resp.answer}\n")


if __name__ == "__main__":
    app()
