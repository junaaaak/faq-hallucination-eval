"""Markdown report and plots for a graded run: results/reports/<name>.md plus figures in results/reports/<name>/."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from faq_eval.grader import Label
from faq_eval.metrics import (
    CategoryRate,
    SettingMetrics,
    category_metrics,
    consistency_metrics,
    paired_test,
    setting_metrics,
)
from faq_eval.runner import RESULTS_DIR, GradedRecord, load_graded

REPORTS_DIR = RESULTS_DIR / "reports"

# Light-mode chart tokens. Model colours are the first three categorical slots, which stay distinguishable
# (including for colour-blind readers) when all three are on screen together.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MODEL_ORDER = ["llama3.2:3b", "qwen2.5:3b", "gemma2:2b"]
PROMPT_ORDER = ["strict", "lenient"]
UNANSWERABLE_CATS = ["out_of_scope", "near_miss", "false_premise", "personal"]
ANSWERABLE_CATS = ["direct", "paraphrased", "multi_fact"]


def _order(values, preferred):
    return [v for v in preferred if v in values] + sorted(v for v in values if v not in preferred)


def _sort_settings(ms: list[SettingMetrics]) -> list[SettingMetrics]:
    models = _order({m.model for m in ms}, MODEL_ORDER)
    prompts = _order({m.prompt for m in ms}, PROMPT_ORDER)
    return sorted(ms, key=lambda m: (models.index(m.model), prompts.index(m.prompt), m.temperature))


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_tradeoff(ms: list[SettingMetrics], path: Path) -> None:
    """Answerable accuracy vs unanswerable refusal rate; each model's strict and lenient prompts joined."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=150, facecolor=SURFACE)
    _style(ax)
    models = _order({m.model for m in ms}, MODEL_ORDER)
    lo = min(min(m.refusal.ci[0], m.accuracy.ci[0]) for m in ms)
    lo = max(0.0, min(0.5, int(lo * 10) / 10))

    # Faint guides: points on the same line have the same balanced score.
    for b in (0.7, 0.8, 0.9):
        if 2 * b - 1.0 >= lo:  # line ends on the right edge
            x0, y0 = 1.0, 2 * b - 1.0
            ax.text(x0, y0, f" balanced {b:.0%}", color=INK_2, fontsize=7.5, va="center", ha="left")
        else:  # line ends on the bottom edge
            x0, y0 = 2 * b - lo, lo
            ax.text(x0, y0 + 0.005, f"balanced {b:.0%} ", color=INK_2, fontsize=7.5, va="bottom", ha="right")
        x1 = max(lo, 2 * b - 1.0)
        ax.plot([x1, x0], [2 * b - x1, y0], color=GRID, linewidth=1, linestyle=(0, (4, 3)), zorder=1)

    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    labels: list[tuple[str, float, float]] = []
    for i, model in enumerate(models):
        color = SERIES[i % len(SERIES)]
        pts = {m.prompt: m for m in ms if m.model == model and m.temperature == min(x.temperature for x in ms)}
        if len(pts) == 2:
            a, b = pts.get("lenient"), pts.get("strict")
            if a and b:
                ax.annotate("", xy=(b.refusal.p, b.accuracy.p), xytext=(a.refusal.p, a.accuracy.p),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=7, shrinkB=7), zorder=2)
        for prompt, m in pts.items():
            x, y = m.refusal.p, m.accuracy.p
            (xl, xh), (yl, yh) = m.refusal.ci, m.accuracy.ci
            ax.plot([xl, xh], [y, y], color=color, linewidth=1, alpha=0.55, zorder=2, solid_capstyle="round")
            ax.plot([x, x], [yl, yh], color=color, linewidth=1, alpha=0.55, zorder=2, solid_capstyle="round")
            filled = prompt == "strict"
            ax.scatter([x], [y], s=70, zorder=3, color=color if filled else SURFACE, edgecolors=color,
                       linewidths=2, )
            labels.append((f"{model.split(':')[0]} {prompt}", x, y))

    _place_labels(fig, ax, labels)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Unanswerable questions correctly refused", color=INK_2, fontsize=9.5)
    ax.set_ylabel("Answerable questions answered correctly", color=INK_2, fontsize=9.5)
    ax.set_title("Strict prompt trades a little accuracy for many more refusals", color=INK, fontsize=11,
                 loc="left", pad=12)
    handles = [Line2D([], [], marker="o", linestyle="", color=SERIES[i % len(SERIES)], markersize=7, label=m)
               for i, m in enumerate(models)]
    handles += [
        Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=7, label="strict prompt"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2,
               markeredgewidth=1.6, markersize=7, label="lenient prompt"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    leg.set_zorder(4)
    ax.text(0.0, -0.13, "Whiskers: Wilson 95% intervals. Arrows go from lenient to strict.",
            transform=ax.transAxes, color=INK_2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _place_labels(fig, ax, labels: list[tuple[str, float, float]]) -> None:
    """Direct labels: try four spots around each point and keep the first that overlaps no point or label."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = [ax.transData.transform((x, y)) for _, x, y in labels]
    taken = [Bbox.from_extents(px - 7, py - 7, px + 7, py + 7) for px, py in pts]
    spots = [((8, 5), "left", "bottom"), ((-8, 5), "right", "bottom"), ((8, -5), "left", "top"),
             ((-8, -5), "right", "top")]
    for text, x, y in labels:
        for k, (offset, ha, va) in enumerate(spots):
            t = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                            color=INK, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85))
            box = t.get_window_extent(renderer)
            if k == len(spots) - 1 or not any(box.overlaps(o) for o in taken):
                taken.append(box)
                break
            t.remove()


def plot_categories(cats: list[CategoryRate], ms: list[SettingMetrics], path: Path) -> None:
    """Heatmap of hallucination rate by unanswerable category for each model and prompt."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rows = [(m.model, m.prompt, m.temperature) for m in _sort_settings(ms)]
    cols = _order({c.category for c in cats if not c.answerable}, UNANSWERABLE_CATS)
    lookup = {(c.model, c.prompt, c.temperature, c.category): c.rate for c in cats if not c.answerable}
    grid = [[lookup[(*r, c)].p if (*r, c) in lookup else float("nan") for c in cols] for r in rows]

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=0.6, aspect="auto")
    # Surface-coloured gaps between cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            rate = lookup.get((*r, c))
            if rate is None:
                continue
            dark = rate.p >= 0.3
            ax.text(j, i, f"{rate.p:.0%}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(cols)), [c.replace("_", " ") for c in cols], color=INK_2, fontsize=9)
    multi_temp = len({r[2] for r in rows}) > 1
    ax.set_yticks(range(len(rows)),
                  [f"{m.split(':')[0]} {p}" + (f" t={t:g}" if multi_temp else "") for m, p, t in rows],
                  color=INK_2, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_title("Hallucination rate on unanswerable questions, by question type", color=INK, fontsize=11,
                 loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _short(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _examples(graded: list[GradedRecord]) -> list[GradedRecord]:
    """One hallucination per unanswerable category, preferring the strict prompt (the harder case)."""
    out = []
    for cat in UNANSWERABLE_CATS:
        hs = [g for g in graded if g.category == cat and g.label == Label.HALLUCINATION]
        # Short answers make the clearest examples.
        hs.sort(key=lambda g: (g.prompt != "strict", len(g.answer), g.question_id))
        if hs:
            out.append(hs[0])
    return out


def plot_temperature(ms: list[SettingMetrics], path: Path) -> None:
    """Hallucination and accuracy against temperature: one column per prompt, one line per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    models = _order({m.model for m in ms}, MODEL_ORDER)
    prompts = _order({m.prompt for m in ms}, PROMPT_ORDER)
    rows = [("hallucination", "Hallucination on unanswerable"), ("accuracy", "Accuracy on answerable")]
    fig, axes = plt.subplots(len(rows), len(prompts), figsize=(3.7 * len(prompts) + 0.9, 6.6), dpi=150,
                             facecolor=SURFACE, sharex=True, sharey="row", squeeze=False)
    for r, (attr, label) in enumerate(rows):
        for c, prompt in enumerate(prompts):
            ax = axes[r][c]
            _style(ax)
            ends = []
            for i, model in enumerate(models):
                pts = sorted((m for m in ms if m.model == model and m.prompt == prompt), key=lambda m: m.temperature)
                if not pts:
                    continue
                color = SERIES[i % len(SERIES)]
                xs = [m.temperature for m in pts]
                rates = [getattr(m, attr) for m in pts]
                ax.fill_between(xs, [x.ci[0] for x in rates], [x.ci[1] for x in rates], color=color, alpha=0.10,
                                linewidth=0)
                ax.plot(xs, [x.p for x in rates], color=color, linewidth=2, zorder=3)
                ax.scatter(xs, [x.p for x in rates], s=26, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4)
                ends.append([xs[-1], rates[-1].p, model.split(":")[0]])
            # Direct labels at the line ends, nudged apart so they don't overlap.
            ends.sort(key=lambda e: e[1])
            for k in range(1, len(ends)):
                ends[k][1] = max(ends[k][1], ends[k - 1][1] + 0.055)
            for x, y, text in ends:
                ax.text(x + 0.05, y, text, color=INK, fontsize=8, va="center", ha="left")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_xlim(min(m.temperature for m in ms) - 0.05, max(m.temperature for m in ms) + 0.45)
            ax.set_xticks(sorted({m.temperature for m in ms}))
            if r == 0:
                ax.set_title(f"{prompt} prompt", color=INK, fontsize=10, loc="left")
            if c == 0:
                ax.set_ylabel(label, color=INK_2, fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("temperature", color=INK_2, fontsize=9)
    fig.suptitle("Temperature and hallucination", color=INK, fontsize=11, x=0.02, ha="left")
    fig.text(0.02, 0.005, "Bands: Wilson 95% intervals. Temperature 0 is one answer per question; higher "
             "temperatures pool three samples.", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def build_report(name: str) -> Path:
    graded_all = load_graded(name)
    if not graded_all:
        raise ValueError(f"no graded results for {name!r}; run `faq-eval grade --name {name}` first")
    ms_all = _sort_settings(setting_metrics(graded_all))
    temps = sorted({m.temperature for m in ms_all})
    t0 = temps[0]
    # Headline sections use the lowest temperature; the temperature section covers the rest.
    graded = [g for g in graded_all if g.temperature == t0]
    ms = [m for m in ms_all if m.temperature == t0]
    cats = category_metrics(graded)
    fig_dir = REPORTS_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(ms, fig_dir / "tradeoff.png")
    plot_categories(cats, ms, fig_dir / "categories.png")
    if len(temps) > 1:
        plot_temperature(ms_all, fig_dir / "temperature.png")

    n_q = len({g.question_id for g in graded_all})
    n_ans = len({g.question_id for g in graded_all if g.answerable})
    methods = Counter(g.method for g in graded_all)
    L: list[str] = []
    w = L.append

    w(f"# Results: `{name}`\n")
    w(f"{len(graded_all)} answers: {n_q} questions ({n_ans} answerable, {n_q - n_ans} unanswerable) × "
      f"{len({m.model for m in ms_all})} models × {len({m.prompt for m in ms_all})} prompts.\n")
    for t in temps:
        s = max(m.samples for m in ms_all if m.temperature == t)
        w(f"- temperature {t:g}: {s} sample{'s' if s > 1 else ''} per question")
    w("")
    w(f"Graded by rules ({methods.get('rule', 0)}) and the LLM judge ({methods.get('judge', 0)})"
      + (f"; {methods['judge_error']} kept the rule label after a judge failure" if methods.get("judge_error") else "")
      + ".\n")
    halfwidths = [(r.ci[1] - r.ci[0]) / 2 for m in ms for r in (m.accuracy, m.hallucination) if r.n]
    per_side = max(m.accuracy.n for m in ms)
    w(f"Intervals are Wilson 95%. With {per_side} answers per side per setting, intervals are "
      f"±{min(halfwidths) * 100:.0f} to ±{max(halfwidths) * 100:.0f} percentage points wide, so differences of a "
      "few points between settings are not meaningful on their own.\n")

    w("## Headline" + (f" (temperature {t0:g})" if len(temps) > 1 else "") + "\n")
    w(f"![Accuracy vs refusal]({name}/tradeoff.png)\n")
    w("| Model | Prompt | Answerable accuracy | False refusals | Unanswerable refused | Hallucination | Balanced |")
    w("|---|---|---|---|---|---|---|")
    for m in ms:
        w(f"| {m.model} | {m.prompt} | {m.accuracy.fmt()} | {m.false_refusal.fmt()} | {m.refusal.fmt()} | "
          f"{m.hallucination.fmt()} | {_pct(m.balanced)} |")
    w("")
    w("Accuracy and false refusals are out of the answerable questions; refusal and hallucination are out of "
      "the unanswerable ones (they add to 100%). Balanced is the mean of accuracy and refusal.\n")

    pairs = []
    for model in _order({m.model for m in ms}, MODEL_ORDER):
        a, b = (model, "lenient", t0), (model, "strict", t0)
        if {a, b} <= {m.key for m in ms}:
            pairs.append((model, paired_test(graded, a, b, Label.HALLUCINATION),
                          paired_test(graded, a, b, Label.CORRECT)))
    if pairs:
        w("## Strict vs lenient prompt\n")
        w("Same questions under both prompts, compared with McNemar's exact test (only questions where the two "
          "prompts disagree count).\n")
        w("| Model | Hallucinations: lenient only / strict only | p | Correct answers: lenient only / strict only | p |")
        w("|---|---|---|---|---|")
        for model, h, c in pairs:
            w(f"| {model} | {h.only_a} / {h.only_b} | {h.p_value:.3g} | {c.only_a} / {c.only_b} | {c.p_value:.3g} |")
        w("")

    if len(temps) > 1:
        cons = {(c.model, c.prompt, c.temperature): c for c in consistency_metrics(graded_all)}
        by_key = {m.key: m for m in ms_all}
        w("## Temperature\n")
        w(f"![Temperature]({name}/temperature.png)\n")
        for title, attr in (("Hallucination rate on unanswerable questions", "hallucination"),
                            ("Accuracy on answerable questions", "accuracy")):
            w(f"{title}:\n")
            w("| Model | Prompt | " + " | ".join(f"t={t:g}" for t in temps) + " |")
            w("|---|---|" + "---|" * len(temps))
            for m in ms:
                cells = [getattr(by_key[(m.model, m.prompt, t)], attr).fmt() if (m.model, m.prompt, t) in by_key
                         else "n/a" for t in temps]
                w(f"| {m.model} | {m.prompt} | " + " | ".join(cells) + " |")
            w("")
        hot = [t for t in temps if t != t0]
        if cons:
            for title, attr in (
                ("Questions where every sample got the same label", "consistent"),
                ("Unanswerable questions hallucinated in at least one sample (the risk for a student who asks a "
                 "few times)", "any_hallucination"),
            ):
                w(f"{title}:\n")
                w("| Model | Prompt | " + " | ".join(f"t={t:g}" for t in hot) + " |")
                w("|---|---|" + "---|" * len(hot))
                for m in ms:
                    ks = [(m.model, m.prompt, t) for t in hot]
                    w(f"| {m.model} | {m.prompt} | "
                      + " | ".join(getattr(cons[k], attr).fmt() if k in cons else "n/a" for k in ks) + " |")
                w("")
        w("Pooling samples treats them as independent, so intervals at temperatures above 0 are somewhat too narrow.\n")

    w("## By question type" + (f" (temperature {t0:g})" if len(temps) > 1 else "") + "\n")
    w(f"![Hallucination by category]({name}/categories.png)\n")
    cols = _order({c.category for c in cats if not c.answerable}, UNANSWERABLE_CATS)
    lookup = {(c.model, c.prompt, c.temperature, c.category): c for c in cats}
    w("Hallucination rate on unanswerable questions:\n")
    w("| Model | Prompt | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    w("|---|---|" + "---|" * len(cols))
    for m in ms:
        cells = [lookup[(*m.key, c)].rate.fmt() if (*m.key, c) in lookup else "n/a" for c in cols]
        w(f"| {m.model} | {m.prompt} | " + " | ".join(cells) + " |")
    w("")
    acols = _order({c.category for c in cats if c.answerable}, ANSWERABLE_CATS)
    w("Accuracy on answerable questions:\n")
    w("| Model | Prompt | " + " | ".join(c.replace("_", " ") for c in acols) + " |")
    w("|---|---|" + "---|" * len(acols))
    for m in ms:
        cells = [lookup[(*m.key, c)].rate.fmt() if (*m.key, c) in lookup else "n/a" for c in acols]
        w(f"| {m.model} | {m.prompt} | " + " | ".join(cells) + " |")
    w("")

    ex = _examples(graded)
    if ex:
        w("## Example hallucinations\n")
        for g in ex:
            w(f"- **{g.category.replace('_', ' ')}** ({g.model}, {g.prompt}, {g.question_id}): *{g.question}*  ")
            w(f"  > {_short(g.answer)}")
        w("")

    from faq_eval import labels

    w("## How far to trust the grader\n")
    a = None
    if labels.label_sheet_path(name).exists():
        try:
            a = labels.agreement(name)
        except ValueError:
            a = None
    if a:
        w(f"{a.n} answers were hand-labelled blind. Rules alone agree with the hand labels on "
          f"{a.rule_accuracy:.1%} ({a.rule_undecided} left undecided); rules plus judge agree on "
          f"{a.final_accuracy:.1%} (Cohen's kappa {a.final_kappa:.2f}). The judge prompt was revised after "
          "seeing these labels (generic advice is no longer counted as a hallucination), so this agreement "
          "is likely optimistic. Details: `results/agreement/`.\n")
    else:
        w("The same grader was checked against 60 hand labels from the main run; see `results/agreement/` "
          "and `results/reports/main.md`. Answers at higher temperatures were not hand-labelled.\n")

    path = REPORTS_DIR / f"{name}.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
