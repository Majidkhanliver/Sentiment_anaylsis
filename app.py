"""Text-only sentiment comparison UI. Run with: streamlit run app.py"""

from html import escape

import pandas as pd
import streamlit as st

from sentiment import MAX_TEXT_LENGTH, MODEL_NAMES, MODEL_PATH, load_models, predict_text


st.set_page_config(page_title="Sentiment Lab", page_icon=":material/compare_arrows:", layout="wide")

# Streamlit sets color-scheme on .stApp; light-dark() follows theme changes immediately.
st.html("""
<style>
.stApp {
    --lab-ink: light-dark(#193247, #e7edf5);
    --lab-muted: light-dark(#5f7080, #a6b4c5);
    --lab-line: light-dark(#dce3e9, #303d50);
    --lab-surface: light-dark(#ffffff, #182231);
    --lab-soft: light-dark(#f0f4f7, #233044);
    --lab-page: light-dark(#f5f7fa, #101722);
    --lab-accent: #087f73;
    background: var(--lab-page);
    color: var(--lab-ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 1120px; padding-top: 2.7rem; padding-bottom: 2rem;
}
[data-testid="stVerticalBlock"] { gap: 1rem; }
.lab-topbar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding-bottom: 20px; border-bottom: 1px solid var(--lab-line);
}
.lab-brand { display: flex; align-items: center; gap: 11px; font-size: 19px; font-weight: 700; letter-spacing: -.5px; }
.lab-logo {
    display: flex; gap: 4px; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 10px; background: #153747;
}
.lab-logo i { display: block; width: 4px; background: #72d7c7; border-radius: 3px; }
.lab-logo i:nth-child(1) { height: 10px; }
.lab-logo i:nth-child(2) { height: 20px; }
.lab-logo i:nth-child(3) { height: 14px; }
.lab-topnote, .lab-muted { color: var(--lab-muted); font-size: 13px; }
.lab-hero { padding: 18px 0 8px; }
.lab-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; color: var(--lab-muted); }
.lab-hero h1 { margin: 0; font-size: clamp(30px, 4vw, 42px); line-height: 1.15; letter-spacing: -1.5px; padding: 10px 0 12px; font-weight: 700; }
.lab-hero p { margin: 0; color: var(--lab-muted); font-size: 16px; line-height: 1.6; }
.st-key-input_panel, .st-key-result_panel {
    background: var(--lab-surface); border: 1px solid var(--lab-line);
    border-radius: 16px; padding: 24px; height: 100%;
    box-shadow: 0 4px 16px rgb(0 0 0 / 2%);
}
.lab-panel-title { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.lab-panel-title h2 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: -.3px; padding: 0; }
.lab-step {
    font-size: 11px; font-weight: 700; color: var(--lab-muted);
    background: var(--lab-soft); padding: 5px 7px; border-radius: 6px;
}
[data-testid="stTextArea"] textarea { font-size: 15px; line-height: 1.65; padding: 14px; }
[data-testid="stTextArea"] [data-baseweb="textarea"] { border-radius: 10px; background: var(--lab-soft); }
[data-testid="stTextArea"] textarea { background: var(--lab-soft); color: var(--lab-ink); }
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within { border-color: var(--lab-accent); }
[data-testid="stButton"] button {
    min-height: 44px; border-radius: 9px; transition: background-color 150ms, border-color 150ms;
}
[data-testid="stButton"] button[kind="primary"] { background: var(--lab-accent); border-color: var(--lab-accent); color: white; }
[data-testid="stButton"] button[kind="primary"]:hover { background: #06685f; border-color: #06685f; }
[data-testid="stButton"] button:focus-visible, textarea:focus-visible {
    outline: 3px solid #219f92; outline-offset: 3px;
}
.st-key-examples { gap: 8px; }
.st-key-examples [data-testid="stButton"] button { padding: 4px 12px; font-size: 13px; }
.st-key-examples button[kind="secondary"] { background: var(--lab-surface); border-color: var(--lab-line); color: var(--lab-ink); }
.st-key-examples button[kind="secondary"]:hover { background: var(--lab-soft); }
.st-key-result_panel [data-testid="stMetricValue"] { font-size: 38px; letter-spacing: -1px; }
.st-key-result_panel [data-testid="stMetricLabel"] { color: var(--lab-muted); }
.lab-empty { text-align: center; padding: 20px 8px 24px; }
.lab-empty-mark {
    width: 62px; height: 62px; margin: 0 auto 18px; border-radius: 18px;
    background: var(--lab-soft); display: flex; align-items: center; justify-content: center; gap: 5px;
}
.lab-empty-mark i { display: block; width: 5px; border-radius: 3px; background: var(--lab-muted); }
.lab-empty-mark i:nth-child(1) { height: 14px; }
.lab-empty-mark i:nth-child(2) { height: 28px; }
.lab-empty-mark i:nth-child(3) { height: 20px; }
.lab-empty h3 { margin: 0; font-size: 19px; padding: 0 0 10px; letter-spacing: -.4px; }
.lab-empty p { color: var(--lab-muted); font-size: 14px; line-height: 1.65; margin: 0 auto; max-width: 280px; }
.lab-empty-footer { margin-top: 24px; font-size: 12px; color: var(--lab-muted); }
.lab-votes { display: grid; gap: 12px; margin: 2px 0 4px; }
.lab-vote { display: grid; grid-template-columns: 65px 1fr 30px; gap: 12px; align-items: center; font-size: 12px; }
.lab-vote strong { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.lab-track { height: 7px; border-radius: 8px; background: var(--lab-soft); overflow: hidden; }
.lab-fill { display: block; height: 100%; border-radius: 8px; }
.lab-fill.positive { background: #138a69; }
.lab-fill.neutral { background: #7b8999; }
.lab-fill.negative { background: #cf5366; }
.lab-comparison-title { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-top: 10px; }
.lab-comparison-title h2 { margin: 0; font-size: 20px; padding: 0; letter-spacing: -.4px; }
.lab-models { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.lab-model {
    background: var(--lab-surface); border: 1px solid var(--lab-line);
    border-radius: 12px; padding: 20px; min-width: 0;
}
.lab-model h3 { margin: 0; font-size: 15px; font-weight: 650; padding: 9px 0 20px; letter-spacing: -.2px; }
.lab-badge { display: inline-flex; gap: 7px; align-items: center; padding: 5px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.lab-badge.positive { background: light-dark(#e8f7ef, #183d34); color: light-dark(#176347, #91d7b4); }
.lab-badge.neutral { background: light-dark(#edf1f6, #293447); color: light-dark(#4c5c70, #c3d2e1); }
.lab-badge.negative { background: light-dark(#fdeef0, #442a36); color: light-dark(#a12e44, #f4a9b5); }
.lab-badge.pending { background: var(--lab-soft); color: var(--lab-muted); }
.lab-badge i { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
[data-testid="stExpander"] { background: var(--lab-surface); border-color: var(--lab-line); border-radius: 12px; }
[data-testid="stExpander"] summary { min-height: 54px; }
.st-key-performance_table { overflow-x: auto; }
.lab-footer { border-top: 1px solid var(--lab-line); padding-top: 18px; color: var(--lab-muted); font-size: 12px; line-height: 1.7; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 3.5rem 1rem 1.5rem; }
    .lab-topnote { display: none; }
    .lab-hero { padding-top: 8px; }
    .lab-hero h1 { letter-spacing: -.9px; }
    .st-key-input_panel, .st-key-result_panel { padding: 20px; }
    .lab-comparison-title { flex-direction: column; gap: 4px; }
    .lab-model { padding: 16px; }
    .lab-model h3 { min-height: 60px; }
}
@media (max-width: 360px) { .lab-models { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { [data-testid="stButton"] button { transition: none; } }
</style>
""")

st.html("""
<div class="lab-topbar">
  <div class="lab-brand"><span class="lab-logo" aria-hidden="true"><i></i><i></i><i></i></span>Sentiment Lab</div>
  <span class="lab-topnote">Two models. A clearer perspective.</span>
</div>
<div class="lab-hero">
  <div class="lab-eyebrow">Text analysis workspace</div>
  <h1>One text. Two perspectives.</h1>
  <p>Explore the sentiment behind your words with Naive Bayes and SVM, and see their collective vote.</p>
</div>
""")


@st.cache_resource(show_spinner="Loading the sentiment models…")
def cached_models(artifact_mtime, model_names):
    # Include the supported models in the cache key when the model list changes.
    return load_models()


try:
    bundle = cached_models(MODEL_PATH.stat().st_mtime_ns, MODEL_NAMES)
except Exception:
    st.error("The saved models could not be loaded. Train them locally, then refresh this page.")
    st.code("python sentiment.py --train", language="bash")
    st.stop()


def clear_result():
    st.session_state.pop("result", None)
    st.session_state.pop("analyzed_text", None)


def set_example(value):
    st.session_state.input_text = value
    clear_result()


input_column, result_column = st.columns([1.12, 1], gap="medium")
with input_column, st.container(key="input_panel", height="stretch"):
    st.html('<div class="lab-panel-title"><span class="lab-step">01</span><h2>Your text</h2></div>')
    text = st.text_area(
        "Text to analyze", height=170, max_chars=MAX_TEXT_LENGTH,
        placeholder="How was your experience? Write or paste a sentence here…",
        key="input_text", on_change=clear_result,
    )
    st.caption("English text · Up to 5,000 characters")
    submitted = st.button(
        "Analyze sentiment", type="primary", width="stretch",
        icon=":material/arrow_forward:", icon_position="right", key="analyze",
    )
    st.caption("Need inspiration? Try an example.")
    with st.container(horizontal=True, key="examples"):
        for label, example in (
            ("Positive", "I really enjoyed the experience and would recommend it."),
            ("Negative", "The product is disappointing and stopped working."),
            ("Mixed", "I love the design, but the service was disappointing."),
        ):
            st.button(label, key=f"example_{label.lower()}", on_click=set_example, args=(example,))
        st.button("Clear", key="clear", type="tertiary", on_click=set_example, args=("",))
    if submitted:
        clear_result()
        try:
            with st.spinner("Comparing Naive Bayes and SVM…"):
                st.session_state.result = predict_text(text, bundle)
                st.session_state.analyzed_text = text
        except ValueError as error:
            st.warning(str(error))

result = st.session_state.get("result")
if result and set(result["predictions"]) != set(MODEL_NAMES):
    clear_result()
    result = None
if st.session_state.get("analyzed_text") != text:
    result = None

with result_column, st.container(key="result_panel", height="stretch"):
    st.html('<div class="lab-panel-title"><span class="lab-step">02</span><h2>Voting result</h2></div>')
    if result:
        st.metric("Selected sentiment", result["sentiment"].capitalize())
        st.write(f"**{result['agreement']} of {len(MODEL_NAMES)} models** predicted **{result['sentiment']}**.")
        bars = "".join(
            f'<div class="lab-vote"><span>{label.capitalize()}</span>'
            f'<div class="lab-track" aria-hidden="true"><span class="lab-fill {label}" '
            f'style="width:{count * 100 / len(MODEL_NAMES):g}%"></span></div>'
            f'<strong>{count}/{len(MODEL_NAMES)}</strong></div>'
            for label, count in result["votes"].items()
        )
        st.html(f'<div class="lab-votes" aria-label="Votes by sentiment">{bars}</div>')
        if result["tie_breaker"]:
            st.info(
                f"Split vote · {result['tie_breaker']} broke the tie. It is the higher-ranked "
                "model on validation data."
            )
        st.caption("Vote agreement is not a probability that the result is correct.")
    else:
        st.html("""
        <div class="lab-empty">
          <div class="lab-empty-mark" aria-hidden="true">
            <i></i><i></i><i></i>
          </div>
          <h3>Your next insight starts here</h3>
          <p>Add your text and select <strong>Analyze sentiment</strong> to see how the models vote.</p>
          <div class="lab-empty-footer">2 model predictions · 1 collective result</div>
        </div>
        """)

st.html("""
<div class="lab-comparison-title">
  <h2 id="each-models-result">Each model’s result</h2>
  <span class="lab-muted">Every model gets one equal vote.</span>
</div>
""")
cards = []
for index, name in enumerate(MODEL_NAMES, start=1):
    sentiment = result["predictions"][name] if result else "pending"
    label = sentiment.capitalize() if result else "Awaiting text"
    cards.append(
        f'<article class="lab-model" aria-label="{escape(name)} prediction">'
        f'<div class="lab-eyebrow">Model {index:02d}</div><h3>{escape(name)}</h3>'
        f'<span class="lab-badge {escape(sentiment, quote=True)}">'
        f'<i aria-hidden="true"></i>{escape(label)}</span></article>'
    )
st.html('<div class="lab-models">' + "".join(cards) + '</div>')

with st.expander("Model performance & voting details"):
    st.write(
        "Each model gets one vote. If both agree, that sentiment wins. If they disagree, "
        "we use the prediction of the higher-ranked model. That ranking comes from validation "
        "macro-F1, with validation accuracy as a secondary measure. With two models, "
        "the voting result always matches the higher-ranked model."
    )
    report = bundle["report"]
    st.caption("Tie-breaking order: " + " → ".join(report["tie_breaking_order"]))
    st.write(f"Held-out evaluation on **{report['data']['test']['usable_rows']:,} texts**:")
    with st.container(key="performance_table"):
        st.table(pd.DataFrame([
            {
                "Model": name,
                "Accuracy": f"{report['test_scores'][name]['accuracy']:.1%}",
                "Macro-F1": f"{report['test_scores'][name]['macro_f1']:.3f}",
                "Negative recall": f"{report['test_scores'][name]['negative_recall']:.1%}",
            }
            for name in MODEL_NAMES
        ]).set_index("Model"))
    voting_scores = report["test_scores"]["Voting ensemble"]
    st.caption(
        f"Combined voting result: accuracy {voting_scores['accuracy']:.1%}, "
        f"macro-F1 {voting_scores['macro_f1']:.3f}. "
        "This combines the two predictions; it is not an additional trained model."
    )
    st.caption(
        "Both models use the same training texts and TF-IDF features. Duplicate text and "
        "conflicting labels are cleaned before splitting; matching test texts are excluded from training. "
        "SVM uses a linear kernel. No model is retrained when you analyze text."
    )

st.html("""
<div class="lab-footer">Built for short English text. Sarcasm, mixed sentiment, and unfamiliar wording can lead to mistakes.<br>
Your text is not saved to disk by the app.</div>
""")
