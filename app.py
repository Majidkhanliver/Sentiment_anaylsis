"""Text-only sentiment comparison UI. Run with: streamlit run app.py"""

import pandas as pd
import streamlit as st

from sentiment import MAX_TEXT_LENGTH, MODEL_NAMES, MODEL_PATH, load_models, predict_text


st.set_page_config(page_title="Sentiment Lab", page_icon=":material/compare_arrows:", layout="centered")

st.caption("SENTIMENT LAB  /  FOUR MODELS, ONE COMPARISON")
st.title("What does your text express?")
st.write("Enter a sentence to compare four sentiment models and see how they vote.")


@st.cache_resource(show_spinner="Loading the sentiment models…")
def cached_models(artifact_mtime):
    return load_models()


try:
    bundle = cached_models(MODEL_PATH.stat().st_mtime_ns)
except Exception:
    st.error("The saved models could not be loaded. Train them locally, then refresh this page.")
    st.code("python sentiment.py --train", language="bash")
    st.stop()


def clear_result():
    st.session_state.pop("result", None)
    st.session_state.pop("analyzed_text", None)


with st.container(border=True):
    text = st.text_area(
        "Text to analyze", height=150, max_chars=MAX_TEXT_LENGTH,
        placeholder="For example: I really enjoyed the experience and would recommend it.",
        help="Best suited to short English text. Maximum 5,000 characters.",
        key="input_text", on_change=clear_result,
    )
    submitted = st.button("Analyze sentiment", type="primary", use_container_width=True)

if submitted:
    clear_result()
    try:
        with st.spinner("Comparing the four models…"):
            st.session_state.result = predict_text(text, bundle)
            st.session_state.analyzed_text = text
    except ValueError as error:
        st.warning(str(error))

result = st.session_state.get("result")
if result and st.session_state.get("analyzed_text") == text:
    st.subheader("Each model's result")
    st.table(pd.DataFrame({
        "Model": list(MODEL_NAMES),
        "Predicted sentiment": [result["predictions"][name].capitalize() for name in MODEL_NAMES],
    }).set_index("Model"))

    with st.container(border=True):
        st.subheader("Final voting result")
        st.metric("Selected sentiment", result["sentiment"].capitalize())
        st.write(f"**{result['agreement']} of 4 models** predicted **{result['sentiment']}**.")
        st.progress(result["agreement"] / 4)
        st.write(" · ".join(f"{label.capitalize()}: **{count}**" for label, count in result["votes"].items()))
        if result["tie_breaker"]:
            st.info(
                f"The vote was tied. {result['tie_breaker']} broke the tie because it is the "
                "highest-ranked model on validation data among those voting for a tied label."
            )
        elif not result["strict_majority"]:
            st.info("This sentiment received the most votes, but did not receive more than half of the votes.")
        st.caption("Model agreement describes the votes. It is not a probability that the result is correct.")
else:
    st.caption("Your results will appear here: Naive Bayes, SVM, Logistic Regression, Random Forest, and the final vote.")

with st.expander("How voting works and how the models performed"):
    st.write(
        "Each model gets one vote. The sentiment with the most votes wins. For a tie, "
        "we use the prediction of the highest-ranked model voting for a tied label. "
        "That ranking comes from validation macro-F1, with validation accuracy as a secondary measure."
    )
    report = bundle["report"]
    st.caption("Tie-breaking order: " + " → ".join(report["tie_breaking_order"]))
    st.write(f"Held-out evaluation on **{report['data']['test']['usable_rows']:,} texts**:")
    st.table(pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{scores['accuracy']:.1%}",
            "Macro-F1": f"{scores['macro_f1']:.3f}",
            "Negative recall": f"{scores['negative_recall']:.1%}",
        }
        for name, scores in report["test_scores"].items()
    ]).set_index("Model"))
    st.caption(
        "All four models use the same training texts and TF-IDF features. Duplicate text and "
        "conflicting labels are cleaned before splitting; matching test texts are excluded from training. "
        "SVM uses a linear kernel. No model is retrained when you analyze text."
    )

st.caption("Trained on social-media text. Sarcasm, mixed sentiment, and unfamiliar wording can lead to mistakes.")
