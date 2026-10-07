"""Train, save, and run Naive Bayes and SVM without changing the notebooks."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import unicodedata

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "sentiment.joblib"
LABELS = ("negative", "neutral", "positive")
MODEL_NAMES = ("Naive Bayes", "SVM")
MAX_TEXT_LENGTH = 5000
SEED = 42


def normalize_text(text):
    """Keep negation and punctuation; normalize the same way for train and predict."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"(?<!\w)@\w+", " usertoken ", text)
    for contraction, expanded in (("can't", "can not"), ("won't", "will not")):
        text = re.sub(rf"\b{contraction}\b", expanded, text)
    text = re.sub(r"n't\b", " not", text)
    return " ".join(text.split())


def load_dataset(path):
    try:
        frame = pd.read_csv(path, usecols=["text", "sentiment"], encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, usecols=["text", "sentiment"], encoding="ISO-8859-1")
    audit = {"raw_rows": len(frame)}
    frame = frame.dropna(subset=["text", "sentiment"]).copy()
    frame["sentiment"] = frame["sentiment"].str.strip().str.lower()
    frame["normalized"] = frame["text"].map(normalize_text)
    frame = frame.loc[frame["normalized"].ne("") & frame["sentiment"].ne("")].copy()
    audit["missing_or_empty_rows"] = audit["raw_rows"] - len(frame)
    if not set(frame["sentiment"]).issubset(LABELS):
        raise ValueError(f"{path.name} contains an unsupported sentiment label.")
    conflicts = frame.groupby("normalized")["sentiment"].nunique()
    conflict_mask = frame["normalized"].isin(conflicts[conflicts > 1].index)
    audit["conflicting_label_rows"] = int(conflict_mask.sum())
    frame = frame.loc[~conflict_mask].copy()
    before = len(frame)
    frame = frame.drop_duplicates("normalized").reset_index(drop=True)
    audit["duplicate_rows"] = before - len(frame)
    return frame, audit


def prepare_data():
    train, train_audit = load_dataset(ROOT / "train.csv")
    test, test_audit = load_dataset(ROOT / "test.csv")
    # Preserve the test set and exclude matching training texts, regardless of label.
    overlap = train["normalized"].isin(test["normalized"])
    train_audit["test_overlap_rows"] = int(overlap.sum())
    train = train.loc[~overlap].reset_index(drop=True)
    for frame, audit in ((train, train_audit), (test, test_audit)):
        audit["usable_rows"] = len(frame)
        audit["class_counts"] = frame["sentiment"].value_counts().to_dict()
        if set(frame["sentiment"]) != set(LABELS):
            raise ValueError("Each dataset must contain all three sentiment classes.")
    return train, test, {"train": train_audit, "test": test_audit}


def new_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True,
        # Retain punctuation and emoji tokens as well as words.
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
    )


def new_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
    }


def vote_predictions(predictions, ranking):
    """Use equal votes; resolve tied labels with the validation-ranked models."""
    if set(predictions) != set(MODEL_NAMES) or any(v not in LABELS for v in predictions.values()):
        raise ValueError("Voting requires one valid prediction from each of Naive Bayes and SVM.")
    if len(ranking) != len(MODEL_NAMES) or set(ranking) != set(MODEL_NAMES):
        raise ValueError("The tie-breaking ranking must contain Naive Bayes and SVM exactly once.")
    counts = Counter(predictions.values())
    top_count = max(counts.values())
    leaders = {label for label, count in counts.items() if count == top_count}
    deciding_model = next(name for name in ranking if predictions[name] in leaders)
    return {
        "sentiment": predictions[deciding_model],
        "votes": {label: counts[label] for label in LABELS},
        "agreement": top_count,
        "tie_breaker": deciding_model if len(leaders) > 1 else None,
        "strict_majority": top_count > len(predictions) / 2,
    }


def score_predictions(actual, predicted):
    report = classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "negative_recall": float(report["negative"]["recall"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def train_models():
    train, test, audit = prepare_data()
    fitting, validation = train_test_split(
        train, test_size=0.2, stratify=train["sentiment"], random_state=SEED,
    )
    vectorizer = new_vectorizer()
    fitting_features = vectorizer.fit_transform(fitting["normalized"])
    validation_features = vectorizer.transform(validation["normalized"])
    validation_scores = {}
    for name, model in new_models().items():
        print(f"Validating {name}...", flush=True)
        model.fit(fitting_features, fitting["sentiment"])
        validation_scores[name] = score_predictions(
            validation["sentiment"], model.predict(validation_features),
        )
    # Test labels never influence the tie-breaking order.
    ranking = sorted(MODEL_NAMES, key=lambda name: (
        -validation_scores[name]["macro_f1"], -validation_scores[name]["accuracy"],
        MODEL_NAMES.index(name),
    ))
    vectorizer = new_vectorizer()
    training_features = vectorizer.fit_transform(train["normalized"])
    test_features = vectorizer.transform(test["normalized"])
    models = new_models()
    predictions, test_scores = {}, {}
    for name, model in models.items():
        print(f"Training and evaluating {name}...", flush=True)
        model.fit(training_features, train["sentiment"])
        predictions[name] = model.predict(test_features)
        test_scores[name] = score_predictions(test["sentiment"], predictions[name])
    ensemble = [
        vote_predictions(dict(zip(MODEL_NAMES, row)), ranking)["sentiment"]
        for row in zip(*(predictions[name] for name in MODEL_NAMES))
    ]
    test_scores["Voting ensemble"] = score_predictions(test["sentiment"], ensemble)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "random_seed": SEED,
        "data": audit,
        "validation_rows": len(validation),
        "validation_scores": validation_scores,
        "tie_breaking_order": ranking,
        "test_scores": test_scores,
        "confusion_matrix_labels": list(LABELS),
    }
    bundle = {"schema_version": 1, "vectorizer": vectorizer, "models": models, "report": report}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    # A failed retraining must not replace an existing usable artifact.
    with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, suffix=".joblib", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        temporary_path.replace(MODEL_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved models to {MODEL_PATH}")
    print(f"Held-out evaluation: {len(test):,} texts")
    for name, scores in test_scores.items():
        print(f"{name:20} accuracy={scores['accuracy']:.3f}  macro-F1={scores['macro_f1']:.3f}")
    return bundle


def load_models():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Train the models first: python sentiment.py --train")
    # This path is generated locally; never load model files supplied by a visitor.
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("schema_version") != 1:
        raise ValueError("The model artifact is outdated. Run: python sentiment.py --train")
    if bundle["report"]["sklearn_version"] != sklearn.__version__:
        raise ValueError("Dependencies changed. Retrain with: python sentiment.py --train")
    if set(bundle["models"]) != set(MODEL_NAMES):
        raise ValueError("The model artifact must contain only Naive Bayes and SVM. Run: python sentiment.py --train")
    return bundle


def predict_text(text, bundle):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text before analyzing.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Use at most {MAX_TEXT_LENGTH:,} characters.")
    if not re.search(r"[a-zA-Z]", text):
        raise ValueError("Enter a sentence in English, including some words.")
    features = bundle["vectorizer"].transform([normalize_text(text)])
    if features.nnz == 0:
        raise ValueError("No familiar words were found. Try a longer sentence in English.")
    predictions = {name: str(model.predict(features)[0]) for name, model in bundle["models"].items()}
    return {"predictions": predictions, **vote_predictions(predictions, bundle["report"]["tie_breaking_order"])}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Train all models and save evaluation results.")
    parser.add_argument("--text", help="Analyze text using the saved models.")
    args = parser.parse_args()
    try:
        if args.train:
            train_models()
        if args.text is not None:
            print(json.dumps(predict_text(args.text, load_models()), indent=2))
        if not args.train and args.text is None:
            parser.print_help()
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
