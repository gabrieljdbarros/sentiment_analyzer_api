# Análise de Sentimento — API + Streamlit

Este projeto realiza análise de sentimento em textos e bases CSV utilizando um modelo de linguagem pré-treinado da Hugging Face.  
A aplicação pode ser executada de duas formas:

1. **Localmente**, com o modelo sendo carregado diretamente no Streamlit (`app_sentiment.py`), ou  
2. **Via API**, em que o Streamlit (`app_client.py`) envia textos para uma API FastAPI (`api_sentiment.py`) que processa as predições.

---

## Estrutura do Projeto

```
.
├── api_sentiment.py      # API FastAPI com modelo Hugging Face
├── app_client.py         # Cliente Streamlit que consome a API
├── app_sentiment.py      # Aplicação Streamlit standalone (modelo local)
├── datasets/             # Pasta para armazenar CSVs de entrada
└── requirements.txt      # Dependências do projeto
```

---

## Requisitos

Antes de iniciar, é necessário ter:

- **Python 3.9+**
- **pip** atualizado
- Recomendado: ambiente virtual (venv ou conda)

### Instalação das dependências

```bash
pip install -r requirements.txt
```

Se não possuir o arquivo `requirements.txt`, as principais dependências são:

```bash
pip install streamlit fastapi uvicorn transformers torch pandas numpy scikit-learn matplotlib
```

---

## Modo 1: Execução Local (sem API)

Executa o Streamlit com o modelo diretamente.

```bash
streamlit run app_sentiment.py
```

A aplicação abrirá automaticamente no navegador, geralmente em:

```
http://localhost:8501
```

### Recursos disponíveis

- Análise de frase única
- Processamento em lote de CSVs
- Exibição de:
  - Distribuição de classes (positivos, neutros, negativos)
  - Série temporal diária (se houver coluna de data)
  - Palavras mais frequentes por sentimento
  - Exemplos por classe
  - Avaliação com rótulos originais (quando disponíveis)
- Download do CSV com os resultados

---

## Modo 2: Execução via API (cliente-servidor)

Neste modo, o modelo roda em uma **API FastAPI** e o Streamlit atua apenas como cliente.

### Etapa 1 — Rodar a API

Em um terminal:

```bash
uvicorn api_sentiment:app --reload
```

A API ficará acessível em:

```
http://127.0.0.1:8000
```

Você pode testar o funcionamento visitando:

```
http://127.0.0.1:8000/docs
```

---

### Etapa 2 — Rodar o cliente Streamlit

Em outro terminal:

```bash
streamlit run app_client.py
```

Acesse no navegador:

```
http://localhost:8501
```

No campo **API URL** da barra lateral, confirme o endereço:
```
http://127.0.0.1:8000
```

---

## Funcionamento Interno

### Fluxo de comunicação (modo API)

1. O usuário envia uma frase ou CSV pelo Streamlit (`app_client.py`).
2. O app envia os textos para a API (`api_sentiment.py`) via requisição POST.
3. A API processa os textos com o modelo `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
4. A API devolve os resultados em formato JSON:
   ```json
   [{"label": "positive", "score": 0.982}, {"label": "negative", "score": 0.941}]
   ```
5. O Streamlit exibe as métricas e gráficos.

### Fluxo (modo local)
O modelo é carregado diretamente no app (`app_sentiment.py`), sem necessidade de API.

---

## Parâmetros Importantes

- **Batch size:** número de textos processados simultaneamente pelo modelo.  
  Valores maiores aumentam a velocidade, mas exigem mais memória.
- **Tamanho da amostra:** número de linhas do CSV a serem analisadas.  
  Controla o volume de dados enviados à API (ou processados localmente).

---

## Saída

O resultado é exibido diretamente no navegador e pode ser baixado em formato CSV contendo:
- Texto original
- Sentimento previsto (`sentiment_label`)
- Confiança da predição (`sentiment_score`)

---

## Personalização

- O modelo pode ser alterado no campo “Hugging Face model id” do sidebar (ex.: `nlptown/bert-base-multilingual-uncased-sentiment`).
- O campo de texto, batch size e tamanho de amostra são configuráveis no menu lateral.
- Caso o CSV contenha uma coluna `date`, o app exibirá automaticamente a série temporal diária de sentimentos.

