# app_client.py
import re
from pathlib import Path
from collections import Counter
import string

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Cliente de Sentimento (via API)", layout="wide")
st.title("Cliente de Análise de Sentimento — envia CSV para a API")

DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", "", str(t))
    t = re.sub(r"@\w+", "", t)
    t = t.replace("#", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_date_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(s.astype(str), errors="coerce")

def parse_sentiment140(df: pd.DataFrame) -> pd.DataFrame:
    """Se o CSV vier sem header (6 colunas), define nomes padrão do Sentiment140."""
    if df is not None and df.shape[1] == 6 and set(df.columns) != {"target","ids","date","flag","user","text"}:
        df.columns = ["target", "ids", "date", "flag", "user", "text"]
    return df

def load_csv_any(path_or_file):
    """
    Lê CSV tentando:
      1) utf-8 sem header
      2) latin-1 sem header
      3) utf-8 com header
      4) latin-1 com header
    e aplica parse do Sentiment140 quando necessário.
    """
    # 1) sem header
    try:
        df = pd.read_csv(path_or_file, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(path_or_file, encoding="latin-1", header=None)
    df = parse_sentiment140(df)

    # 2) se ainda não houver 'text', tenta com header
    has_file_like = hasattr(path_or_file, "read")
    if "text" not in df.columns:
        if has_file_like:
            try:
                path_or_file.seek(0)
            except Exception:
                pass
            try:
                df_try = pd.read_csv(path_or_file, encoding="utf-8")
            except UnicodeDecodeError:
                if has_file_like:
                    try:
                        path_or_file.seek(0)
                    except Exception:
                        pass
                df_try = pd.read_csv(path_or_file, encoding="latin-1")
            if isinstance(df_try, pd.DataFrame):
                return df_try
        else:
            # é um Path -> tenta ler com header inferido
            try:
                df_try = pd.read_csv(path_or_file, encoding="utf-8")
            except UnicodeDecodeError:
                df_try = pd.read_csv(path_or_file, encoding="latin-1")
            if isinstance(df_try, pd.DataFrame):
                return df_try

    return df

def normalize_label(lbl: str) -> str:
    l = str(lbl).lower()
    if "star" in l:  # p/ modelos tipo nlptown (1-5 stars)
        if "1" in l or "2" in l: return "negative"
        if "3" in l: return "neutral"
        return "positive"
    return l

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

def post_predict_any(api_url, records, timeout=120):
    """
    Tenta três formatos comuns de contrato de /predict:
      A) lista de objetos [{"text": "..."}]
      B) {"texts": ["...", "..."]}
      C) lista de strings ["...", "..."]
    Retorna o JSON já convertido (list).
    """
    # A
    try:
        r = requests.post(f"{api_url}/predict", json=records, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        pass

    # B
    try:
        texts = [r["text"] for r in records]
        r2 = requests.post(f"{api_url}/predict", json={"texts": texts}, timeout=timeout)
        r2.raise_for_status()
        return r2.json()
    except requests.HTTPError:
        pass

    # C
    texts = [r["text"] for r in records]
    r3 = requests.post(f"{api_url}/predict", json=texts, timeout=timeout)
    r3.raise_for_status()
    return r3.json()

# --------------------- Sidebar: Config API ---------------------
st.sidebar.header("Config API")
api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")

if st.sidebar.button("Ping /health"):
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        st.sidebar.success(f"/health → {r.status_code}: {r.text}")
    except Exception as e:
        st.sidebar.error(f"Falha no ping: {e}")

# --------------------- Sidebar: CSV & envio ---------------------
st.sidebar.header("Entrada em CSV")
uploaded = st.sidebar.file_uploader("Drag and drop file here", type=["csv"], help="Limite 200MB por arquivo")

csvs = sorted([p.name for p in DATA_DIR.glob("*.csv")])
choice = st.sidebar.selectbox("Ou escolha um CSV em datasets/", ["(nenhum)"] + csvs)

text_col_default = "text"
text_col = st.sidebar.text_input("Coluna de texto", value=text_col_default)

# NOVO: coluna de data (opcional)
date_col = st.sidebar.text_input(
    "Coluna de data (opcional, ex.: 'date' no Sentiment140)",
    value="date"
)

chunk = st.sidebar.slider("Tamanho do lote de envio para API", 50, 5000, 500, 50)

sample_n = st.sidebar.number_input(
    "Tamanho da amostra a enviar para a API",
    min_value=100,
    max_value=200_000,
    value=10_000,
    step=1000,
    help="Número de linhas do CSV que serão ENVIADAS para a API (amostra aleatória)."
)

st.sidebar.selectbox("Codificação do CSV", ["auto (utf-8→latin-1)"], index=0, disabled=True)

# --------------------- Frase única via API ---------------------
st.subheader("Teste rápido — Frase única (API)")
single_text = st.text_input("Digite uma frase para classificar:")

col_run1, col_clear1 = st.columns([1,1])
with col_run1:
    if st.button("Analisar frase (API)"):
        if single_text.strip():
            try:
                resp = post_predict_any(api_url, [{"text": single_text.strip()}], timeout=30)
                # resp esperado: [{"label": "...", "score": 0.99}]
                if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], dict):
                    label = normalize_label(resp[0].get("label", ""))
                    score = float(resp[0].get("score", 0.0))
                    st.success(f"Sentimento: **{label}** | score: {score:.3f}")
                else:
                    st.warning(f"Resposta inesperada da API: {resp}")
            except Exception as e:
                st.error(f"Falha ao chamar a API: {e}")
        else:
            st.warning("Digite uma frase primeiro.")
with col_clear1:
    if st.button("Limpar frase"):
        st.experimental_rerun()

st.markdown("---")

# --------------------- Carregar CSV ---------------------
st.subheader("Carregamento do CSV")
df = None
if uploaded is not None:
    df = load_csv_any(uploaded)
elif choice != "(nenhum)":
    df = load_csv_any(DATA_DIR / choice)

if df is None:
    st.info("Envie um CSV ou selecione um arquivo em datasets/ para continuar.")
    st.stop()

if text_col not in df.columns:
    st.error(f"A coluna '{text_col}' não existe. Colunas detectadas: {list(df.columns)}")
    st.stop()

st.success(f"{len(df):,} linhas carregadas.".replace(",", "."))
st.dataframe(df.head(10), use_container_width=True)

# --------------------- Enviar para API ---------------------
st.markdown("---")
if st.button("Rodar análise no CSV via API", type="primary"):
    # 1) Seleciona/limpa texto e inclui data (se existir)
    cols = [text_col]
    if date_col and date_col in df.columns:
        cols.append(date_col)

    df_to_send = df[cols].rename(columns={text_col: "text"}).copy()
    df_to_send["text"] = df_to_send["text"].astype(str).map(clean_text)

    # cria _date (apenas dia) se houver data
    if date_col in df_to_send.columns:
        dts = parse_date_series(df_to_send[date_col])
        df_to_send["_date"] = dts.dt.date
    else:
        df_to_send["_date"] = pd.NaT

    # 2) Aplica amostra
    if len(df_to_send) > sample_n:
        df_to_send = df_to_send.sample(sample_n, random_state=42).reset_index(drop=True)

    st.success(f"Enviando **{len(df_to_send):,}** linhas para a API…".replace(",", "."))
    progress = st.progress(0)
    preds = []
    n = len(df_to_send)

    # 3) Envia em lotes para /predict
    try:
        for i in range(0, n, chunk):
            batch_df = df_to_send.iloc[i:i+chunk]
            payload = batch_df[["text"]].to_dict(orient="records")  # só envia o texto
            resp = post_predict_any(api_url, payload, timeout=120)
            preds.extend(resp)
            progress.progress(min((i+chunk)/n, 1.0))
    except requests.HTTPError as e:
        st.error(f"Erro HTTP ao chamar a API: {e} | Resposta: {getattr(e, 'response', None) and e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Falha ao enviar/receber da API: {e}")
        st.stop()

    if len(preds) != len(df_to_send):
        st.warning(f"API retornou {len(preds)} predições para {len(df_to_send)} linhas. Alinhando pelo menor comprimento.")
        m = min(len(preds), len(df_to_send))
        preds = preds[:m]
        df_to_send = df_to_send.iloc[:m].reset_index(drop=True)

    # 4) Anexa resultados ao dataframe da amostra
    df_pred = df_to_send.copy()
    df_pred["sentiment_label"] = [normalize_label(p.get("label", "")) for p in preds]
    df_pred["sentiment_score"] = [float(p.get("score", 0.0)) for p in preds]

    st.subheader("Amostra com predições")
    st.dataframe(df_pred.head(10), use_container_width=True)

    # 5) Resumo por classe
    st.subheader("Distribuição por classe (amostra enviada)")
    order = ["positive", "neutral", "negative"]
    counts = df_pred["sentiment_label"].value_counts().reindex(order).fillna(0).astype(int)
    perc = (counts / max(counts.sum(), 1) * 100).round(2)
    c1, c2, c3 = st.columns(3)
    c1.metric("Positivos", f"{counts.get('positive',0)} ({perc.get('positive',0)}%)")
    c2.metric("Neutros",   f"{counts.get('neutral',0)} ({perc.get('neutral',0)}%)")
    c3.metric("Negativos", f"{counts.get('negative',0)} ({perc.get('negative',0)}%)")
    st.bar_chart(counts.rename("linhas por classe"))

    # 6) Série temporal (por dia) usando df_pred
    st.subheader("Evolução diária (amostra enviada)")
    if "_date" in df_pred.columns and df_pred["_date"].notna().any():
        grp = (
            df_pred.dropna(subset=["_date"])
                   .groupby(["_date", "sentiment_label"])
                   .size()
                   .unstack(fill_value=0)
                   .sort_index()
        )
        if not grp.empty:
            st.line_chart(grp)
        else:
            st.caption("Sem dados suficientes para montar a série temporal.")
    else:
        st.caption("Nenhuma coluna de data válida foi informada/detectada; gráfico diário indisponível.")

    # 7) Palavras mais comuns (por classe)
    st.subheader("Palavras mais comuns por classe")
    n_top = st.slider("Número de palavras mais frequentes", 5, 30, 10)
    cols_top = st.columns(3)
    for i, cls in enumerate(["positive", "neutral", "negative"]):
        with cols_top[i]:
            st.markdown(f"**{cls.capitalize()}**")
            subset = df_pred[df_pred["sentiment_label"] == cls]["text"].dropna().astype(str)
            words = Counter([w for txt in subset for w in tokenize(txt)])
            if words:
                top_words = dict(words.most_common(n_top))
                st.bar_chart(pd.Series(top_words))
            else:
                st.caption("Sem textos nessa classe.")

    # 8) Exemplos por classe
    st.subheader("Exemplos por classe")
    n_show = st.slider("Exemplos por classe", 3, 20, 5)
    cols_show = ["text","sentiment_label","sentiment_score"]
    for cls in ["positive","neutral","negative"]:
        st.markdown(f"**{cls.capitalize()}**")
        ex = df_pred[df_pred["sentiment_label"]==cls].nlargest(n_show, "sentiment_score")[cols_show]
        if ex.empty:
            st.caption("Sem exemplos nessa classe.")
        else:
            st.dataframe(ex, use_container_width=True)

    # 9) Download da amostra com predições
    base_name = "(upload)" if uploaded is not None else choice.rsplit(".", 1)[0]
    out_name = f"{base_name}_pred_api_sample_{len(df_pred)}.csv"
    st.download_button(
        "Baixar CSV (amostra + predições)",
        df_pred.to_csv(index=False).encode("utf-8"),
        out_name,
        "text/csv"
    )

else:
    st.info("Ajuste **amostra** e **lote** na barra lateral e clique em **Rodar análise no CSV via API**.")
