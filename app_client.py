# app_client.py
import json
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Cliente de Sentimento (via API)", layout="wide")
st.title("Cliente de Análise de Sentimento — envia CSV para a API")

DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# --------------------- Utils ---------------------
def clean_text(t: str) -> str:
    t = re.sub(r"http\S+", "", str(t))
    t = re.sub(r"@\w+", "", t)
    t = t.replace("#", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_sentiment140(df: pd.DataFrame) -> pd.DataFrame:
    """Se o CSV vier sem header (6 colunas), define nomes padrão do Sentiment140."""
    if df is not None and df.shape[1] == 6 and set(df.columns) != {"target","ids","date","flag","user","text"}:
        df.columns = ["target", "ids", "date", "flag", "user", "text"]
    return df

def load_csv_any(path_or_file):
    """Lê CSV em utf-8; se falhar, tenta latin-1. Tenta com e sem header."""
    try:
        df = pd.read_csv(path_or_file, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(path_or_file, encoding="latin-1", header=None)
    # Se tiver um header real, pandas já terá detectado; se não, aplica Sentiment140
    df = parse_sentiment140(df)
    # Se mesmo assim não houver 'text', tente reler com header=True (caso o dataset tenha cabeçalho real)
    if "text" not in df.columns and hasattr(path_or_file, "read"):
        path_or_file.seek(0)
        try:
            df_try = pd.read_csv(path_or_file, encoding="utf-8")
        except UnicodeDecodeError:
            path_or_file.seek(0)
            df_try = pd.read_csv(path_or_file, encoding="latin-1")
        if isinstance(df_try, pd.DataFrame) and "text" in df_try.columns:
            return df_try
    return df

def normalize_label(lbl: str) -> str:
    l = lbl.lower()
    if "star" in l:  # p/ modelos tipo nlptown (1-5 stars)
        if "1" in l or "2" in l: return "negative"
        if "3" in l: return "neutral"
        return "positive"
    return l

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

chunk = st.sidebar.slider("Tamanho do lote de envio para API", 50, 5000, 500, 50)

sample_n = st.sidebar.slider(
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

def post_predict_any(api_url, records, timeout=30):
    import requests
    # A) lista de objetos [{"text": "..."}]
    try:
        r = requests.post(f"{api_url}/predict", json=records, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        # B) {"texts": ["...", "..."]}
        try:
            texts = [r["text"] for r in records]
            r = requests.post(f"{api_url}/predict", json={"texts": texts}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            # C) lista de strings ["...", "..."]
            texts = [r["text"] for r in records]
            r = requests.post(f"{api_url}/predict", json=texts, timeout=timeout)
            r.raise_for_status()
            return r.json()

with col_run1:
    if st.button("Analisar frase (API)"):
        if single_text.strip():
            try:
                # envia como lista de 1 item, compatível com /predict da API de referência
                resp = post_predict_any(api_url, [{"text": single_text.strip()}], timeout=30)
                # resp esperado: [{"label": "...", "score": 0.99}]
                if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], dict):
                    label = str(resp[0].get("label", "")).lower()
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
    # 1) Seleciona e limpa a coluna de texto
    df_to_send = df[[text_col]].rename(columns={text_col: "text"}).copy()
    df_to_send["text"] = df_to_send["text"].astype(str).map(clean_text)

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
            payload = batch_df.to_dict(orient="records")  # [{"text": "..."}...]
            r = requests.post(f"{api_url}/predict", json=payload, timeout=120)
            r.raise_for_status()
            resp = r.json()

            # resp esperado: [{"label":"positive","score":0.98}, ...]
            preds.extend(resp)
            progress.progress(min((i+chunk)/n, 1.0))
    except requests.HTTPError as e:
        st.error(f"Erro HTTP ao chamar a API: {e} | Resposta: {getattr(e, 'response', None) and e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Falha ao enviar/receber da API: {e}")
        st.stop()

    if len(preds) != len(df_to_send):
        st.warning(f"API retornou {len(preds)} predições para {len(df_to_send)} linhas. Tentando alinhar pelo menor comprimento.")
        m = min(len(preds), len(df_to_send))
        preds = preds[:m]
        df_to_send = df_to_send.iloc[:m].reset_index(drop=True)

    # 4) Anexa resultados ao dataframe original da amostra
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

    # 6) Download
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
