# app_sentiment.py
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import string
import streamlit as st
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
)

st.set_page_config(page_title="Análise de Sentimento (CSV + Texto)", layout="wide")
st.title("Análise de Sentimento — CSV + Frase única")

DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------- Utils ----------------------
def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", "", str(t))
    t = re.sub(r"@\w+", "", t)
    t = t.replace("#", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def star_to_polarity(label: str) -> str:
    l = label.lower()
    if "1" in l or "2" in l: return "negative"
    if "3" in l: return "neutral"
    return "positive"

@st.cache_resource(show_spinner=False)
def load_hf_pipeline(model_id: str):
    # Força tokenizer "slow" para modelos com SentencePiece (ex.: XLM-R)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return TextClassificationPipeline(model=model, tokenizer=tok)
    except Exception:
        return pipeline("sentiment-analysis", model=model_id)

def normalize_label(lbl: str) -> str:
    lbl = lbl.lower()
    if "star" in lbl:
        return star_to_polarity(lbl)
    return lbl  # "positive"/"neutral"/"negative"

def parse_sentiment140(df: pd.DataFrame) -> pd.DataFrame:
    # Se o CSV veio sem header (6 colunas), aplica nomes padrão
    if df.shape[1] == 6 and set(df.columns) != {"target","ids","date","flag","user","text"}:
        df.columns = ["target", "ids", "date", "flag", "user", "text"]
    return df

def parse_date_series(s: pd.Series) -> pd.Series:
    # Sentiment140 tem datas no formato string; tenta várias heurísticas
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(s.astype(str), errors="coerce")

# ---------------------- Sidebar ----------------------
st.sidebar.header("Entrada de dados (CSV)")
csvs = sorted([p.name for p in DATA_DIR.glob("*.csv")])
choice = st.sidebar.selectbox("Escolha um CSV em datasets/", ["(nenhum)"] + csvs)
uploaded = st.sidebar.file_uploader("Ou envie um CSV", type=["csv"])

st.sidebar.header("Configuração do modelo")
model_id = st.sidebar.text_input(
    "Hugging Face model id",
    value="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    help="Ex.: cardiffnlp/twitter-xlm-roberta-base-sentiment (tweets, multilíngue) ou nlptown/bert-base-multilingual-uncased-sentiment (1–5 stars)."
)

# Processamento
default_text_col = "text"
text_col = st.sidebar.text_input("Coluna de texto no CSV", value=default_text_col)
batch = st.sidebar.slider("Batch size", 16, 256, 64, step=16)

sample_n = st.sidebar.number_input(
    "Tamanho da amostra a enviar para a API",
    min_value=1,
    value=10000,
    step=1000,
    help="Número de linhas do CSV que serão ENVIADAS para a API (amostra aleatória)."
)


# ---------------------- Frase única ----------------------
st.subheader("Teste rápido — Frase única")
single_text = st.text_input("Digite uma frase para classificar:")
c_run, c_clear = st.columns([1,1])
with c_run:
    if st.button("Analisar frase"):
        if single_text.strip():
            pipe = load_hf_pipeline(model_id)
            pred = pipe([clean_text(single_text)])[0]
            lbl = normalize_label(pred["label"])
            st.success(f"Sentimento: **{lbl}** | score: {pred['score']:.3f}")
        else:
            st.warning("Digite uma frase primeiro.")
with c_clear:
    if st.button("Limpar frase"):
        st.experimental_rerun()

st.markdown("---")

# ---------------------- CSV: carregar ----------------------
st.subheader("Classificação em CSV")
df = None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded, encoding="latin-1", header=None)
elif choice != "(nenhum)":
    try:
        df = pd.read_csv(DATA_DIR / choice, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_DIR / choice, encoding="latin-1", header=None)

if df is None:
    st.info("Selecione ou envie um CSV para rodar a análise em lote.")
    st.stop()

df = parse_sentiment140(df)

if text_col not in df.columns:
    st.error(f"A coluna '{text_col}' não existe. Colunas: {list(df.columns)}")
    st.stop()

st.success(f"{len(df):,} linhas carregadas.".replace(",", "."))
st.dataframe(df.head(10), use_container_width=True)

# Amostragem para não travar
if len(df) > sample_n:
    st.caption(f"Usando amostra de {sample_n} linhas para a primeira análise (configure no sidebar).")
    df = df.sample(sample_n, random_state=42).reset_index(drop=True)

# ---------------------- Rodar predição no CSV ----------------------
if st.button("Rodar análise no CSV", type="primary"):
    pipe = load_hf_pipeline(model_id)

    texts = df[text_col].astype(str).map(clean_text).tolist()
    labels, scores = [], []

    progress = st.progress(0)
    total = len(texts)
    for i in range(0, total, batch):
        preds = pipe(texts[i:i+batch])
        for p in preds:
            labels.append(normalize_label(p["label"]))
            scores.append(float(p["score"]))
        progress.progress(min((i+batch)/total, 1.0))

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    # --------- Resumo por classe ---------
    st.subheader("Distribuição por classe")
    order = ["positive", "neutral", "negative"]
    counts = df["sentiment_label"].value_counts().reindex(order).fillna(0).astype(int)
    perc = (counts / max(counts.sum(), 1) * 100).round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Positivos", f"{counts.get('positive',0)} ({perc.get('positive',0)}%)")
    c2.metric("Neutros",   f"{counts.get('neutral',0)} ({perc.get('neutral',0)}%)")
    c3.metric("Negativos", f"{counts.get('negative',0)} ({perc.get('negative',0)}%)")
    st.bar_chart(counts.rename("linhas por classe"))

    # --------- Série temporal (se houver 'date') ---------
    if "date" in df.columns:
        st.subheader("Evolução diária (se houver data)")
        dts = parse_date_series(df["date"])
        grp = df.assign(_date=dts.dt.date).dropna(subset=["_date"]).groupby(["_date","sentiment_label"]).size().unstack(fill_value=0)
        if not grp.empty:
            st.line_chart(grp)
        else:
            st.caption("Datas não reconhecidas — pulando série temporal.")

  # --------- Palavras mais comuns ---------
    st.subheader("Palavras mais comuns por classe")

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    n_top = st.slider("Número de palavras mais frequentes", 5, 30, 10)
    cols = st.columns(3)
    for i, cls in enumerate(["positive", "neutral", "negative"]):
        with cols[i]:
            st.markdown(f"**{cls.capitalize()}**")
            subset = df[df["sentiment_label"] == cls]["text"].dropna().astype(str)
            words = Counter([w for txt in subset for w in tokenize(txt)])
            if words:
                top_words = dict(words.most_common(n_top))
                st.bar_chart(pd.Series(top_words))
            else:
                st.caption("Sem textos nessa classe.")   # --------- Exemplos por classe ---------
    st.subheader("Exemplos por classe")
    n_show = st.slider("Exemplos por classe", 3, 20, 5)
    cols = ["text","sentiment_label","sentiment_score"]
    for cls in ["positive","neutral","negative"]:
        st.markdown(f"**{cls.capitalize()}**")
        ex = df[df["sentiment_label"]==cls].nlargest(n_show, "sentiment_score")[cols]
        if ex.empty:
            st.caption("Sem exemplos nessa classe.")
        else:
            st.dataframe(ex, use_container_width=True)

    # --------- Avaliação (se tiver rótulo) ---------
    with st.expander("Avaliar com rótulos do dataset (ex.: Sentiment140 'target')"):
        label_col = st.text_input("Coluna de rótulo", value="target")
        if label_col in df.columns:
            mapping = {"0":"negative", "4":"positive", 0:"negative", 4:"positive"}
            truth = df[label_col].map(mapping)
            if truth.notna().any():
                from sklearn.metrics import classification_report
                mask = truth.notna()
                st.code(classification_report(truth[mask], df.loc[mask, "sentiment_label"], digits=3))
            else:
                st.info("Não foi possível mapear os rótulos desta coluna.")
        else:
            st.info("Informe o nome correto da coluna de rótulo para avaliar.")

    # --------- Export ---------
    base_name = "(upload)" if uploaded is not None else choice.rsplit(".", 1)[0]
    out_name = f"{base_name}_scored.csv"
    st.download_button(
        "Baixar CSV com sentimento",
        df.to_csv(index=False).encode("utf-8"),
        out_name,
        "text/csv"
    )
else:
    st.info("Clique em **Rodar análise no CSV** para processar a amostra.")
