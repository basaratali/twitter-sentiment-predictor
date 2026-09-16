import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Twitter sentiment predictor", page_icon="🐦")

MODELS_DIR = Path(__file__).parent / "models"

REQUIRED_FILES = [
    "tfidf_vectorizer.pkl",
    "label_encoder.pkl",
    "logistic_regression.pkl",
    "naive_bayes.pkl",
    "linear_svm.pkl",
    "simple_rnn.keras",
]

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
    "Linear SVM": "linear_svm",
    "SimpleRNN": "simple_rnn",
}


def clean_tweet(text):
    # Must match the cleaning used to train every model in the notebook exactly —
    # any drift here would silently feed each model out-of-distribution input.
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pkl(filename):
    with open(MODELS_DIR / filename, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        return None, missing

    artifacts = {
        "tfidf": load_pkl("tfidf_vectorizer.pkl"),
        "label_encoder": load_pkl("label_encoder.pkl"),
        "logistic_regression": load_pkl("logistic_regression.pkl"),
        "naive_bayes": load_pkl("naive_bayes.pkl"),
        "linear_svm": load_pkl("linear_svm.pkl"),
        "simple_rnn": tf.keras.models.load_model(MODELS_DIR / "simple_rnn.keras"),
    }
    return artifacts, []


st.title("Twitter sentiment predictor")
st.caption(
    "Pick a model trained on the Twitter Entity Sentiment Analysis dataset, enter text like a "
    "tweet, and see its predicted sentiment. \"Irrelevant\" means the text mentions the entity "
    "but isn't really expressing an opinion about it."
)

artifacts, missing = load_artifacts()

if artifacts is None:
    st.error(
        "Model files not found in the `models/` folder next to this app.\n\n"
        "Missing: " + ", ".join(missing) + "\n\n"
        "Run the notebook's \"Part 4 — Save the models\" section, download the resulting "
        "zip, and extract its contents into the `models/` folder here."
    )
    st.stop()

label_encoder = artifacts["label_encoder"]
class_names = label_encoder.classes_

with st.form("predict_form", border=False):
    tweet_text = st.text_area(
        "Tweet text",
        placeholder="e.g. This new update completely ruined the game, so disappointed...",
        height=100,
    )
    model_choice = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    submitted = st.form_submit_button("Predict sentiment", icon=":material/send:")

if submitted:
    if not tweet_text.strip():
        st.warning("Enter some text first.")
        st.stop()

    cleaned = clean_tweet(tweet_text)
    if not cleaned:
        st.warning(
            "That text has no usable words left after cleaning (letters only) — try something else."
        )
        st.stop()

    model_key = MODEL_OPTIONS[model_choice]

    with st.container(border=True):
        st.markdown(f"**{model_choice}**")

        if model_key == "simple_rnn":
            # tf.constant (not np.array) — numpy infers a fixed-width unicode dtype
            # (<U...) for a Python string, which Keras's input validation rejects;
            # tf.constant produces a proper dtype=tf.string tensor instead
            text_tensor = tf.constant([cleaned])
            probs = artifacts["simple_rnn"].predict(text_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            st.metric("Prediction", class_names[pred_idx])
            st.caption(f"Confidence: {probs[pred_idx]:.1%}")

        elif model_key == "linear_svm":
            tfidf_vec = artifacts["tfidf"].transform([cleaned])
            margins = artifacts["linear_svm"].decision_function(tfidf_vec)[0]
            pred_idx = int(np.argmax(margins))
            st.metric("Prediction", class_names[pred_idx])
            st.caption(f"Decision margin: {margins[pred_idx]:.2f} (not a probability)")

        else:
            tfidf_vec = artifacts["tfidf"].transform([cleaned])
            probs = artifacts[model_key].predict_proba(tfidf_vec)[0]
            pred_idx = int(np.argmax(probs))
            st.metric("Prediction", class_names[pred_idx])
            st.caption(f"Confidence: {probs[pred_idx]:.1%}")
