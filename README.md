# Sentiment Lab

A text-box UI that shows predictions from **Naive Bayes, Linear SVM, Logistic
Regression, and Random Forest**, followed by their equal-weight voting result.
The existing notebooks and CSV files are preserved.

## Run locally

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python sentiment.py --train
python -m streamlit run app.py
```

Training runs separately from the UI. It creates `artifacts/sentiment.joblib`
and `artifacts/sentiment.json` (evaluation metrics, confusion matrices, data
cleaning counts, and validation rankings). Subsequent app starts reuse the saved
models. Retrain after changing the model code, training data, or scikit-learn
version. Only load locally generated model artifacts.

The artifact uses `joblib`, a pickle-based persistence format with convenient
compression for NumPy-heavy scikit-learn objects. It stores the fitted vectorizer,
all four models, and evaluation metadata together. Plain Python `pickle` would
also work; changing serialization format does not change predictions or accuracy.

For a terminal prediction:

```bash
python sentiment.py --text "I really enjoyed this!"
```

## Evaluation and voting

- All models train on the larger `train.csv`; `sen1.csv` remains available for
  the original SVM notebook but is not mixed into the new training data.
- Only `text` and `sentiment` are used. The CSV encodings are detected with a
  UTF-8 read followed by a Latin-1 fallback.
- Missing text/labels, normalized duplicate text, and groups with conflicting
  labels are removed in memory. Matching test texts are excluded from training.
- Normalization handles whitespace, casing, apostrophes, URLs, mentions, and
  negative contractions. Negation, stopwords, punctuation, and emoji are retained
  as candidate features. Vocabulary limits may still exclude rare tokens.
- A stratified 80/20 split of cleaned training data determines the tie-breaking
  order by macro-F1, then accuracy. An exact ranking tie uses the displayed model
  order. TF-IDF is fitted only on the fitting partition during validation.
- Final models and TF-IDF are refitted on the full cleaned training set and
  evaluated against cleaned `test.csv`. Test labels do not select voting weights,
  model settings, or tie-breaking order. Settings are fixed baselines, not a
  hyperparameter search.
- Every model gets one vote. A unique highest count wins, including a 2–1–1
  plurality. For tied counts, the highest-ranked model supporting a tied label
  decides. The UI explains both ties and results without a strict majority.
- Vote agreement is not prediction confidence. The performance panel compares
  the ensemble against each individual model; voting need not improve accuracy.

## Scope

The UI accepts one English text at a time (up to 5,000 characters), with no file
upload. Inputs are not saved to disk. It is intended for short social-media-style
text; sarcasm, mixed sentiment, other languages, and longer passages are not
reliably covered by this dataset.

The original notebooks keep their historical outputs and machine-specific paths.
They are independent of the new app. No test files are required to run the app.
