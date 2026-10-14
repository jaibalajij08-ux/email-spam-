"""Train Naive Bayes, Logistic Regression and SVM spam classifiers; save the best one."""
import io, re, urllib.request
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "spam.csv"
SMS_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data() -> pd.DataFrame:
    # 1) Kaggle file: data/spam.csv (columns v1, v2)
    if DATA.exists():
        df = pd.read_csv(DATA, encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
        print("Loaded Kaggle dataset from data/spam.csv")
        return df
    # 2) Auto-download the SMS Spam Collection
    try:
        raw = urllib.request.urlopen(SMS_URL, timeout=15).read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["label", "text"])
        print("Downloaded SMS Spam Collection")
        return df
    except Exception as e:
        print(f"Download failed ({e}); using tiny built-in sample data")
    # 3) Fallback sample
    return pd.read_csv(ROOT / "data" / "sample.csv")


def main():
    df = load_data().dropna()
    df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    df["clean"] = df["text"].apply(clean)
    print(f"{len(df)} messages | spam ratio {df['label'].mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": LinearSVC(class_weight="balanced"),
    }
    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
                         ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"model": name,
                        "accuracy": accuracy_score(y_test, pred),
                        "precision": precision_score(y_test, pred, zero_division=0),
                        "recall": recall_score(y_test, pred, zero_division=0),
                        "f1": f1_score(y_test, pred, zero_division=0)})
        fitted[name] = (pipe, pred)

    res = pd.DataFrame(results).sort_values("f1", ascending=False).round(4)
    print("\n", res.to_string(index=False))
    best = res.iloc[0]["model"]
    pipe, pred = fitted[best]
    print(f"\nBest model: {best}\n")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], zero_division=0))

    joblib.dump(pipe, ROOT / "spam_model.joblib")
    res.to_csv(ROOT / "results.csv", index=False)

    # Confusion matrix
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix - {best}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(ROOT / "confusion_matrix.png", dpi=120); plt.close()

    # Top spam keywords (from a Logistic Regression fit on all data)
    lr = Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(df["clean"], df["label"])
    vocab = np.array(lr["tfidf"].get_feature_names_out())
    coefs = lr["clf"].coef_[0]
    top = np.argsort(coefs)[-15:]
    plt.figure(figsize=(6, 4.5))
    plt.barh(vocab[top], coefs[top], color="crimson")
    plt.title("Top spam-indicating keywords"); plt.tight_layout()
    plt.savefig(ROOT / "spam_keywords.png", dpi=120); plt.close()
    print("Saved: spam_model.joblib, results.csv, confusion_matrix.png, spam_keywords.png")


if __name__ == "__main__":
    main()
