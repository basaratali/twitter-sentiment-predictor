# Twitter sentiment predictor

A Streamlit app that predicts sentiment (Positive / Negative / Neutral / Irrelevant) on
tweet-like text, using a model you pick from a dropdown. Trained on the
[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
dataset (~74K tweets, 4 classes).

Four models are included, so you can compare a classical ML approach against a deep learning
one on the same input:

| Model | Validation accuracy |
| --- | --- |
| Linear SVM | 95.8% |
| Logistic Regression | 91.8% |
| SimpleRNN | 90.8% |
| Naive Bayes | 80.5% |

All trained on TF-IDF features (unigrams + bigrams), except the SimpleRNN, which uses a
`TextVectorization` layer with a learned embedding instead.

## Run it

Requires Python 3.9–3.12 (TensorFlow does not yet support the very latest Python releases —
this was built and tested on 3.11):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

The trained model files are already included under `models/` — no training step needed, it
runs immediately.

## How it works

- **Logistic Regression / Naive Bayes / Linear SVM** — scikit-learn models on TF-IDF features,
  saved as standard `.pkl` files (Python's `pickle` module).
- **SimpleRNN** — a Keras model (`Embedding` → `SimpleRNN` → `Dropout` → `Dense(softmax)`) with
  the `TextVectorization` layer built into the model itself, so the exact vocabulary used for
  training travels with the saved model. Saved as `.keras`, not `.pkl` — TensorFlow models hold
  internal state (the computational graph, `tf.Variable`s) that plain `pickle` can't serialize
  correctly; `.keras` is Keras's own format and the correct equivalent for a model of this type.
- Text is cleaned the same way for every model before prediction (lowercased, URLs/@mentions
  stripped, letters only) — the exact same cleaning used to train them, since any mismatch here
  would silently feed each model out-of-distribution input.

Linear SVM doesn't produce a calibrated probability (`predict_proba` isn't available for
`LinearSVC`), so its result shows a decision margin instead of a confidence percentage — shown
that way deliberately rather than faking a probability number.

## Project structure

```
app.py              Streamlit app
requirements.txt
models/              Trained model files (tfidf_vectorizer.pkl, label_encoder.pkl,
                      logistic_regression.pkl, naive_bayes.pkl, linear_svm.pkl, simple_rnn.keras)
```

The training code (data cleaning, model comparison, and how these files were produced) lives in
a separate notebook, not included in this repository.
