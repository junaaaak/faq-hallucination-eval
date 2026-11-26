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

## Status

Work in progress: temperature sweep and Streamlit demo still to come.
