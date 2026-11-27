"""Streamlit demo: ask the FAQ bot live, and explore the recorded runs.

Start it with `faq-eval demo` (or `streamlit run src/faq_eval/app.py`).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from faq_eval import demo, runner
from faq_eval.dataset import Question, load_fact_sheet, load_questions
from faq_eval.grader import Label
from faq_eval.judge import DEFAULT_JUDGE_MODEL
from faq_eval.report import REPORTS_DIR

MODELS = ["llama3.2:3b", "qwen2.5:3b", "gemma2:2b"]
PROMPTS = ["strict", "lenient"]
CATEGORY_TEXT = {
    "direct": "Direct", "paraphrased": "Paraphrased", "multi_fact": "Multi-fact",
    "out_of_scope": "Out of scope", "near_miss": "Near miss", "false_premise": "False premise",
    "personal": "Personal",
}
CUSTOM = "Type my own question…"
DEFAULT_QUESTION = "u031"

st.set_page_config(page_title="FAQ hallucination eval", layout="wide")


@st.cache_data
def questions() -> list[Question]:
    return load_questions()


@st.cache_data
def fact_sheet() -> str:
    return load_fact_sheet()


@st.cache_data
def run_data(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = runner.load_graded(name)
    return demo.records_frame(records), demo.metrics_frame(records)


@st.cache_data(ttl=30)
def installed_models() -> list[str] | None:
    """Models Ollama has pulled, or None if Ollama isn't reachable."""
    try:
        import ollama

        resp = ollama.Client().list()
        models = resp["models"] if isinstance(resp, dict) else resp.models
        return [m["model"] if isinstance(m, dict) else m.model for m in models]
    except Exception:
        return None


def label_badge(label: Label) -> str:
    color = "green" if label in demo.GOOD_LABELS else "orange" if label == Label.UNCERTAIN else "red"
    return f":{color}-badge[{demo.LABEL_TEXT[label]}]"


def question_text(q: Question) -> str:
    kind = "answerable" if q.answerable else "unanswerable"
    return f"{q.id} · {CATEGORY_TEXT[q.category.value]} ({kind}) · {q.question}"


st.title("College FAQ bot: hallucination eval")
st.caption(
    "Small local models answer questions about a fictional college from a fixed fact sheet. Some questions "
    "can't be answered from the sheet, so the right reply is to say so. Anything else is made up."
)

runs = demo.available_runs()
tab_ask, tab_results, tab_browse, tab_sheet = st.tabs(["Ask the bot", "Results", "Browse answers", "Fact sheet"])

# --- Ask the bot -----------------------------------------------------------------------------------
with tab_ask:
    installed = installed_models()
    if installed is None:
        st.warning("Ollama isn't reachable, so live answers are off. Start Ollama and refresh; the other tabs "
                   "work without it.")
    qs = questions()
    left, right = st.columns([3, 2], gap="large")
    with left:
        # Default to a near-miss question the lenient prompt tends to answer with made-up companies.
        default = next((i + 1 for i, q in enumerate(qs) if q.id == DEFAULT_QUESTION), 1)
        choice = st.selectbox("Question", [CUSTOM, *qs], index=default,
                              format_func=lambda q: q if isinstance(q, str) else question_text(q))
        if choice == CUSTOM:
            question: Question | str = st.text_input("Your question", placeholder="When does the library close?")
            st.caption("Your own questions aren't graded, since there's no reference answer; you'll see "
                       "whether the bot refused.")
        else:
            question = choice
            if question.answerable:
                facts = " and ".join(" / ".join(g) for g in question.required_facts)
                st.caption(f"Answerable. A correct answer mentions: {facts}")
            else:
                st.caption("Not answerable from the fact sheet. The bot should say it doesn't know, or correct "
                           "the question's assumption.")
    with right:
        model = st.selectbox("Model", MODELS)
        prompts = st.pills("System prompt", PROMPTS, selection_mode="multi", default=PROMPTS,
                           help="strict: answer only from the sheet and use a fixed refusal otherwise. "
                                "lenient: just told to use the sheet.")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
        use_judge = st.toggle(f"Grade with the LLM judge ({DEFAULT_JUDGE_MODEL})", value=False,
                              help="Rules grade most answers. The judge handles the ones rules can't decide; "
                                   "it's slower because Ollama has to swap models.")

    missing = [m for m in [model, DEFAULT_JUDGE_MODEL if use_judge else None] if m and installed
               and m not in installed]
    if missing:
        st.info(f"Not pulled yet: {', '.join(missing)}. Run `ollama pull {missing[0]}` first.")
    ready = installed is not None and not missing and bool(prompts) and bool(question)
    if st.button("Ask", type="primary", disabled=not ready):
        cols = st.columns(len(prompts))
        for col, prompt in zip(cols, sorted(prompts, key=PROMPTS.index)):
            with col:
                st.subheader(f"{prompt} prompt")
                with st.spinner("Thinking…"):
                    try:
                        res = demo.ask(question, model, prompt, temperature,
                                       judge_model=DEFAULT_JUDGE_MODEL if use_judge else None)
                    except Exception as exc:
                        st.error(f"Ollama call failed: {exc}")
                        continue
                if res.grade:
                    st.markdown(f"{label_badge(res.grade.label)} &nbsp; {res.grade.reason} "
                                f"_({res.grade.method})_")
                else:
                    st.markdown(":blue-badge[Refused]" if res.refusal else ":gray-badge[Answered]")
                with st.container(border=True, height=320):
                    st.markdown(res.response.answer or "_(empty answer)_")
                st.caption(f"{res.response.latency_s:.1f} s · {res.response.completion_tokens or '?'} tokens"
                           + (" · cut off at the token limit" if res.response.truncated else ""))

    if isinstance(question, Question) and runs:
        st.divider()
        st.markdown(f"**How this question went in the recorded `{runs[0]}` run**")
        df, _ = run_data(runs[0])
        st.dataframe(demo.recorded_answers(df, question.id), hide_index=True, use_container_width=True)

# --- Results ---------------------------------------------------------------------------------------
pct = st.column_config.NumberColumn(format="%.0f%%")

with tab_results:
    if not runs:
        st.info("No graded runs yet. Run `faq-eval run` and `faq-eval grade` first.")
    else:
        name = st.selectbox("Run", runs, key="results_run")
        df, mdf = run_data(name)
        temps = sorted(mdf["temperature"].unique())
        st.markdown(f"{len(df):,} graded answers · {df['question_id'].nunique()} questions · "
                    f"temperature{'s' if len(temps) > 1 else ''} {', '.join(f'{t:g}' for t in temps)}. "
                    "Intervals are Wilson 95%.")
        t = st.select_slider("Temperature", temps, value=temps[0]) if len(temps) > 1 else temps[0]
        view = mdf[mdf["temperature"] == t].drop(columns=["temperature"])
        view = view.sort_values(["model", "prompt"], key=lambda s: s.map(
            lambda v: (MODELS + PROMPTS).index(v) if v in MODELS + PROMPTS else 99))
        rate_cols = ["accuracy", "hallucination", "false refusal", "balanced"]
        view[rate_cols] = view[rate_cols] * 100
        st.dataframe(view, hide_index=True, use_container_width=True, column_config={c: pct for c in rate_cols})
        st.caption("Accuracy and false refusals are out of answerable questions; hallucination is out of "
                   "unanswerable ones. Balanced is the mean of accuracy and the refusal rate.")
        fig_dir = REPORTS_DIR / name
        for fig, caption in (("temperature.png", "Temperature"), ("tradeoff.png", "Accuracy vs refusal"),
                             ("categories.png", "Hallucination by question type")):
            if (fig_dir / fig).exists():
                st.image(str(fig_dir / fig), caption=caption, width=900)
        st.caption(f"Full write-up: `results/reports/{name}.md` (regenerate with `faq-eval report --name {name}`).")

# --- Browse answers --------------------------------------------------------------------------------
with tab_browse:
    if runs:
        name = st.selectbox("Run", runs, key="browse_run")
        df, _ = run_data(name)
        c1, c2, c3, c4, c5 = st.columns(5)
        f_model = c1.multiselect("Model", sorted(df["model"].unique()))
        f_prompt = c2.multiselect("Prompt", PROMPTS)
        f_temp = c3.multiselect("Temperature", sorted(df["temperature"].unique()))
        f_cat = c4.multiselect("Category", list(CATEGORY_TEXT), format_func=CATEGORY_TEXT.get)
        f_label = c5.multiselect("Label", [l.value for l in Label if l != Label.UNCERTAIN],
                                 default=["hallucination"], format_func=lambda v: demo.LABEL_TEXT[Label(v)])
        sub = demo.filter_frame(df, model=f_model, prompt=f_prompt, temperature=f_temp, category=f_cat,
                                label=f_label)
        st.caption(f"{len(sub):,} of {len(df):,} answers. Select a row to read the full answer.")
        sub = sub.reset_index(drop=True)
        event = st.dataframe(
            sub[["question_id", "model", "prompt", "temperature", "category", "question", "answer", "label",
                 "method", "reason"]],
            hide_index=True, use_container_width=True, height=420, on_select="rerun",
            selection_mode="single-row", key=f"browse_table_{name}",
            column_config={"answer": st.column_config.TextColumn(width="large"),
                           "question": st.column_config.TextColumn(width="medium")},
        )
        rows = event.selection.rows
        if rows and rows[0] < len(sub):
            r = sub.iloc[rows[0]]
            st.markdown(f"**{r['question_id']} · {r['question']}**")
            st.markdown(f"{label_badge(Label(r['label']))} &nbsp; {r['model']}, {r['prompt']} prompt, "
                        f"temperature {r['temperature']:g} · graded by {r['method']}: "
                        f"{r['reason'].removeprefix('judge: ')}")
            with st.container(border=True):
                st.markdown(r["answer"])

# --- Fact sheet ------------------------------------------------------------------------------------
with tab_sheet:
    st.caption("Everything the bot is allowed to know. The college is fictional, so an answer that isn't "
               "supported here is made up.")
    st.markdown(fact_sheet())
