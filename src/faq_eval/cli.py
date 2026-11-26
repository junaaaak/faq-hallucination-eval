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
    """Run a handful of questions (one per kind) and print the answers with their grades."""
    from faq_eval.bot import BotConfig, FAQBot
    from faq_eval.dataset import load_questions
    from faq_eval.grader import grade

    by_id = {q.id: q for q in load_questions()}
    bot = FAQBot(BotConfig(model=model, prompt=prompt))
    for qid in SMOKE_IDS:
        q = by_id[qid]
        resp = bot.ask(q.question)
        expected = q.required_facts or q.premise_corrections or "refusal"
        typer.echo(f"[{q.id} | {q.category.value} | {resp.latency_s:.1f}s] {q.question}")
        typer.echo(f"  expected: {expected}")
        typer.echo(f"  answer:   {resp.answer}")
        g = grade(q, resp.answer)
        typer.echo(f"  grade:    {g.label.value} ({g.reason})\n")


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@app.command("run")
def run_cmd(
    name: str = typer.Option(..., help="Run name; answers go to results/runs/<name>.jsonl."),
    models: str = typer.Option("llama3.2:3b", help="Comma-separated Ollama models."),
    prompts: str = typer.Option("strict,lenient", help="Comma-separated prompt variants."),
    temperatures: str = typer.Option("0", help="Comma-separated temperatures."),
    samples: int = typer.Option(1, help="Samples per question per setting (seed = sample index)."),
) -> None:
    """Ask every question under every setting. Resumes if interrupted."""
    from faq_eval.runner import run

    def progress(i: int, total: int, rec) -> None:
        if i == 1 or i % 20 == 0 or i == total:
            typer.echo(f"  {i}/{total}  {rec.model} {rec.prompt} t={rec.temperature} s={rec.sample}")

    path = run(name, _split(models), _split(prompts), [float(t) for t in _split(temperatures)], samples,
               on_progress=progress)
    typer.echo(f"answers: {path}")


@app.command("grade")
def grade_cmd(
    name: str = typer.Option(..., help="Run name to grade."),
    judge_model: str = typer.Option("qwen2.5:7b", help="Ollama model for the judge."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Rules only; leave undecided answers as uncertain."),
    rejudge: bool = typer.Option(False, "--rejudge", help="Ignore saved judge verdicts (after changing the judge)."),
) -> None:
    """Grade a run with the rules, then the LLM judge for answers the rules can't decide."""
    from collections import Counter

    from faq_eval.judge import Judge
    from faq_eval.runner import grade_run, load_graded

    judge = None if no_judge else Judge(model=judge_model)

    def progress(i: int, total: int, g) -> None:
        if g.method == "judge":
            typer.echo(f"  judged {g.question_id} ({g.prompt}): {g.rule_label.value} -> {g.label.value}")

    path = grade_run(name, judge=judge, on_progress=progress, rejudge=rejudge)
    graded = load_graded(name)
    for key in sorted({(g.model, g.prompt, g.temperature) for g in graded}):
        counts = Counter(g.label.value for g in graded if (g.model, g.prompt, g.temperature) == key)
        typer.echo(f"{key[0]:<18} {key[1]:<8} t={key[2]:<4} {dict(sorted(counts.items()))}")
    typer.echo(f"graded: {path}")


@app.command("label-sheet")
def label_sheet_cmd(
    name: str = typer.Option(..., help="Graded run to sample from."),
    n: int = typer.Option(60, help="Number of answers to label."),
    overwrite: bool = typer.Option(False, help="Replace an existing sheet (loses any labels in it)."),
) -> None:
    """Write a blind CSV of answers to hand-label."""
    from faq_eval.labels import HUMAN_LABELS, make_label_sheet

    path = make_label_sheet(name, n=n, overwrite=overwrite)
    typer.echo(f"label sheet: {path}")
    typer.echo(f"fill the human_label column with one of: {', '.join(HUMAN_LABELS)}")


@app.command("agreement")
def agreement_cmd(
    name: str = typer.Option(..., help="Run whose label sheet has been filled in."),
    partial_as: str = typer.Option("error", help="How to count 'partial' hand labels: error or ok."),
) -> None:
    """Compare hand labels with the rule-only and rule+judge labels."""
    from faq_eval.labels import agreement

    a = agreement(name, partial_as=partial_as)
    typer.echo(f"'partial' counted as:        {partial_as}")
    typer.echo(f"labelled answers:            {a.n}")
    typer.echo(f"rules only agree:            {a.rule_accuracy:.1%}  ({a.rule_undecided} left uncertain)")
    typer.echo(f"rules + judge agree:         {a.final_accuracy:.1%}  (Cohen's kappa {a.final_kappa:.2f})")
    typer.echo(f"agree on 'is it an error?':  {a.error_accuracy:.1%}")
    typer.echo("\nconfusion (human -> grader):")
    for h, row in a.confusion.items():
        typer.echo(f"  {h:<16} {row}")
    if a.disagreements:
        typer.echo("\ndisagreements:")
        for d in a.disagreements:
            typer.echo(
                f"  #{d['item']} {d['question_id']}: human={d['raw']} ({d['human']}) grader={d['grader']} ({d['method']})"
            )



@app.command("report")
def report_cmd(name: str = typer.Option(..., help="Graded run to report on.")) -> None:
    """Write results/reports/<name>.md with tables, confidence intervals and plots."""
    from faq_eval.metrics import setting_metrics
    from faq_eval.report import build_report
    from faq_eval.runner import load_graded

    path = build_report(name)
    for m in setting_metrics(load_graded(name)):
        typer.echo(
            f"{m.model:<14} {m.prompt:<8} t={m.temperature:<4g} acc {m.accuracy.fmt()}  "
            f"halluc {m.hallucination.fmt()}  balanced {m.balanced:.0%}"
        )
    typer.echo(f"report: {path}")

if __name__ == "__main__":
    app()
