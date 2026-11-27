# FAQ Hallucination Eval

A small, local evaluation of how often LLMs hallucinate when acting as a college FAQ bot.

The bot gets a fixed fact sheet for a **fictional** college and is asked questions, some of which
cannot be answered from the sheet. Because the college is made up, any answer not supported by the
sheet is a hallucination by definition.

## What is measured

- **Accuracy** on answerable questions
- **Refusal rate** on unanswerable questions (saying "I don't know" is the correct answer)
- **False refusal rate** and **hallucination rate**, so a bot that refuses everything doesn't look perfect
- **Stretch:** how temperature changes the hallucination rate

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull gemma2:2b
```

## Project layout

```
data/            fact sheet and question set
src/faq_eval/    bot, grader, runner, metrics, CLI
tests/           unit tests
results/         run outputs, reports, plots
```

## Results

The main run (3 models × 2 prompts × 120 questions, temperature 0) is written up in
[results/reports/main.md](results/reports/main.md), with Wilson 95% intervals and plots.
Regenerate it with:

```bash
faq-eval report --name main
```

The temperature sweep (temperatures 0.3, 0.7, 1.0 and 1.3, three samples per question, 9,360 answers in all)
is in [results/reports/sweep.md](results/reports/sweep.md). In short: over this range, temperature made no
detectable difference to hallucination or accuracy (none of 12 per-setting trend tests reached p < 0.05).
The system prompt mattered far more. At higher temperatures, though, the same question can get different
answers. With the lenient prompt, 33–58% of unanswerable questions were hallucinated in at least one of
three samples, against a per-answer rate of 19–40%.

The temperature-0 answers in the sweep were copied from the main run (`results/runs/sweep.notes.md`).

```bash
faq-eval run --name sweep --models llama3.2:3b,qwen2.5:3b,gemma2:2b --prompts strict,lenient \
  --temperatures 0.3,0.7,1.0,1.3 --samples 3
faq-eval grade --name sweep
faq-eval report --name sweep
```

## Status

Work in progress: Streamlit demo still to come.
