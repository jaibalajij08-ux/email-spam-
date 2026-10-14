import re
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model():
    path = ROOT / "spam_model.joblib"
    if not path.exists():
        import train
        train.main()
    return joblib.load(path)


model = load_model()
st.title("📧 Email Spam Classifier")
st.caption("Naive Bayes / Logistic Regression / SVM with TF-IDF - best model auto-selected")

text = st.text_area("Paste an email or message:", height=180,
                    placeholder="Congratulations! You won a FREE prize. Click here to claim...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        x = [clean(text)]
        clf = model["clf"]
        if hasattr(clf, "predict_proba"):
            p = float(model.predict_proba(x)[0][1])
        else:
            p = float(1 / (1 + 2.718281828 ** (-2 * model.decision_function(x)[0])))
        if model.predict(x)[0] == 1:
            st.error(f"🚨 SPAM  (confidence {p:.1%})")
        else:
            st.success(f"✅ HAM / Legitimate  (spam score {p:.1%})")

with st.expander("Model performance & spam keywords"):
    for f, cap in [("confusion_matrix.png", "Confusion matrix"), ("spam_keywords.png", "Top spam keywords")]:
        if (ROOT / f).exists():
            st.image(str(ROOT / f), caption=cap)
