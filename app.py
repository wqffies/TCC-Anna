"""
RAG - Regulamentacoes de IA em Universidades Federais
"""

import os
import re
from collections import Counter
from pathlib import Path
from typing import List

import fitz
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import chromadb

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

PDF_DIR = Path(__file__).parent / "pdfs"

LLM_MODEL       = "llama-3.3-70b-versatile"
EMBEDDING_MODEL  = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 400
TOP_K_EACH    = 10

TOP_K_ESPECIFICA  = 5
TOP_K_COMPARATIVA = 8
TOP_K_GERAL       = 12

ALIASES_UNIVERSIDADE = {
    "Unifesp": "UNIFESP", "unifesp": "UNIFESP", "sao paulo": "UNIFESP", "são paulo": "UNIFESP",
    "ufpb": "UFPB", "paraiba": "UFPB", "paraíba": "UFPB",
    "ufc": "UFC", "ceara": "UFC", "ceára": "UFC",
    "ufba": "UFBA", "bahia": "UFBA",
    "ufdpar": "UFDPAR", "delta do parnaiba": "UFDPAR",
    "uff": "UFF", "fluminense": "UFF",
    "ufg": "UFG", "goias": "UFG", "goiás": "UFG",
    "ufma": "UFMA", "maranhao": "UFMA", "maranhão": "UFMA",
    "ufmg": "UFMG", "minas gerais": "UFMG",
    "ufms": "UFMS", "mato grosso": "UFMS",
    "ufop": "UFOP", "ouro preto": "UFOP",
    "ufrj": "UFRJ", "rio de janeiro": "UFRJ",
    "ufu": "UFU", "uberlandia": "UFU", "uberlândia": "UFU",
    "unifal": "UNIFAL MG", "alfenas": "UNIFAL MG",
    "unir": "UNIR", "rondonia": "UNIR", "rondônia": "UNIR",
    "uffrj": "UFFRJ",
}

SYSTEM_PROMPT = (
    "Voce e um assistente especializado em regulamentacoes sobre uso de "
    "Inteligencia Artificial em universidades federais brasileiras.\n\n"
    "Responda com base EXCLUSIVAMENTE nos trechos de documentos fornecidos. "
    "Seja claro, objetivo e sempre cite de qual universidade e documento veio a informacao. "
    "Para perguntas comparativas, organize a resposta por universidade. "
    "Se a informacao nao estiver nos trechos, diga explicitamente que nao encontrou nos documentos. "
    "Responda sempre em portugues brasileiro."
)

EXEMPLOS_PERGUNTAS = [
    "A UFPB permite uso de IA em trabalhos acadêmicos?",
    "Compare as políticas da UFMG e da UFRJ sobre IA em provas.",
    "Quais universidades proíbem completamente o uso de IA?",
    "O que a UNIFESP diz sobre plágio com IA?",
    "Como a UFC regulamenta o uso de IA na pós-graduação?",
    "Existe punição para uso indevido de IA na UFBA?",
]

# ---------------------------------------------------------------------------
# CSS personalizado
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── Importar fontes ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Remover barra superior do Streamlit ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
footer { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Variáveis de cor ── */
:root {
    --bg-primary:    #0e1117;
    --bg-secondary:  #161b27;
    --bg-card:       #1c2333;
    --bg-hover:      #232d42;
    --accent:        #4f8ef7;
    --accent-soft:   rgba(79, 142, 247, 0.12);
    --accent-glow:   rgba(79, 142, 247, 0.25);
    --gold:          #f0a500;
    --gold-soft:     rgba(240, 165, 0, 0.10);
    --text-primary:  #e8eaf0;
    --text-secondary:#8b93a8;
    --text-muted:    #555f73;
    --border:        rgba(255,255,255,0.07);
    --border-accent: rgba(79, 142, 247, 0.30);
    --radius:        12px;
    --radius-sm:     8px;
}

/* ── Fundo geral ── */
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"] > section:first-child {
    background-color: var(--bg-primary) !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── Área de conteúdo principal ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 860px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Texto global ── */
body, p, span, div, li, label {
    font-family: 'Sora', sans-serif !important;
    color: var(--text-primary);
}

/* ── Título principal ── */
h1 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-bottom: 0.25rem !important;
}

/* ── Subtítulos ── */
h2, h3 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

/* ── Caption / texto pequeno ── */
[data-testid="stCaptionContainer"] p,
.stCaption, small {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
}

/* ── Botões de exemplo de perguntas ── */
.pergunta-btn {
    display: block;
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.4rem;
    color: var(--text-primary);
    font-family: 'Sora', sans-serif;
    font-size: 0.78rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.18s ease;
    line-height: 1.45;
}
.pergunta-btn:hover {
    background: var(--bg-hover);
    border-color: var(--border-accent);
    color: var(--accent);
    transform: translateX(3px);
}

/* ── Botões de universidade ── */
.uni-btn {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.65rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s ease;
    letter-spacing: 0.03em;
}
.uni-btn:hover {
    background: var(--accent-soft);
    border-color: var(--border-accent);
    color: var(--accent);
}

/* ── Streamlit buttons (para reutilizar o st.button) ── */
[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    padding: 0.55rem 0.85rem !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    white-space: normal !important;
    line-height: 1.45 !important;
    height: auto !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-accent) !important;
    color: var(--accent) !important;
    transform: translateX(3px) !important;
}

/* ── Botões de universidade (área principal) ── */
.main .stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    padding: 0.25rem 0.65rem !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.03em !important;
    height: auto !important;
    line-height: 1.4 !important;
    margin: 0.15rem 0.15rem 0.15rem 0 !important;
}
.main .stButton > button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--border-accent) !important;
    color: var(--accent) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.85rem 1.1rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--border-accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Mensagens do chat ── */
[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: var(--accent-soft) !important;
    border-color: var(--border-accent) !important;
}

/* ── Expander (fontes) ── */
[data-testid="stExpander"] {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--text-primary) !important;
}

/* ── Divisor ── */
hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ── Badge de tipo de consulta ── */
.tipo-badge {
    display: inline-block;
    background: var(--gold-soft);
    border: 1px solid rgba(240,165,0,0.25);
    border-radius: 20px;
    padding: 0.18rem 0.6rem;
    font-size: 0.7rem;
    color: var(--gold);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-right: 0.4rem;
}
.chunks-badge {
    display: inline-block;
    background: var(--accent-soft);
    border: 1px solid var(--border-accent);
    border-radius: 20px;
    padding: 0.18rem 0.6rem;
    font-size: 0.7rem;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── Header decorativo ── */
.app-header {
    border-left: 3px solid var(--accent);
    padding-left: 1rem;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    margin-bottom: 0.2rem !important;
}
.app-header .subtitle {
    color: var(--text-secondary);
    font-size: 0.84rem;
    line-height: 1.5;
}

/* ── Seção de universidades ── */
.uni-section-label {
    color: var(--text-muted);
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Sidebar label ── */
.sidebar-label {
    color: var(--text-muted);
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
</style>
"""

# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def inferir_universidade(nome_arquivo: str) -> str:
    nome_norm = nome_arquivo.lower().replace("-", "_").replace(" ", "_")
    for alias in sorted(ALIASES_UNIVERSIDADE, key=len, reverse=True):
        pattern = r"\b" + re.escape(alias.replace(" ", "_")) + r"\b"
        if re.search(pattern, nome_norm):
            return ALIASES_UNIVERSIDADE[alias]
    return os.path.splitext(nome_arquivo)[0].replace("_", " ").replace("-", " ").upper()


def extrair_universidades_da_query(query: str) -> list:
    query_norm = query.lower().strip()
    encontradas = []
    for alias in sorted(ALIASES_UNIVERSIDADE, key=len, reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, query_norm):
            nome = ALIASES_UNIVERSIDADE[alias]
            if nome not in encontradas:
                encontradas.append(nome)
    return encontradas


def classificar_query(universidades: list) -> tuple:
    if len(universidades) >= 2:
        return "comparativa", TOP_K_COMPARATIVA
    if len(universidades) == 1:
        return "especifica", TOP_K_ESPECIFICA
    return "geral", TOP_K_GERAL

# ---------------------------------------------------------------------------
# Extracao de PDF
# ---------------------------------------------------------------------------

def extrair_texto_pdf(caminho_pdf: Path) -> list:
    doc = fitz.open(str(caminho_pdf))
    universidade = inferir_universidade(caminho_pdf.name)
    paginas = []
    for num_pagina in range(len(doc)):
        texto = doc[num_pagina].get_text().strip()
        if len(texto) > 50:
            paginas.append({
                "texto":        texto,
                "pagina":       num_pagina + 1,
                "arquivo":      caminho_pdf.name,
                "universidade": universidade,
            })
    doc.close()
    return paginas

# ---------------------------------------------------------------------------
# Pipeline (carregado uma unica vez via cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def construir_pipeline():
    groq_api_key = st.secrets["GROQ_API_KEY"]

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, groq_api_key=groq_api_key)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        st.error(
            f"Nenhum PDF encontrado na pasta '{PDF_DIR.name}/'. "
            "Verifique se os arquivos estão no repositório."
        )
        st.stop()

    todas_paginas = []
    for pdf in pdfs:
        todas_paginas.extend(extrair_texto_pdf(pdf))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documentos = [
        Document(
            page_content=f"Regulamentacao de IA da {p['universidade']}: {p['texto']}",
            metadata={"arquivo": p["arquivo"], "universidade": p["universidade"], "pagina": p["pagina"]},
        )
        for p in todas_paginas
    ]

    chunks = splitter.split_documents(documentos)

    chroma_client = chromadb.EphemeralClient()
    chroma_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=chroma_client,
        collection_name="regulamentacoes_ia",
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_EACH

    contagem = Counter(c.metadata["universidade"] for c in chunks)
    return chroma_store, bm25, reranker, llm, contagem, len(pdfs)

# ---------------------------------------------------------------------------
# Retrieval e geracao
# ---------------------------------------------------------------------------

def rrf_multi(listas: list, k: int = 60) -> List[Document]:
    scores, doc_map = {}, {}
    for lista in listas:
        for rank, doc in enumerate(lista):
            chave = doc.page_content[:120]
            scores[chave]  = scores.get(chave, 0) + 1 / (k + rank + 1)
            doc_map[chave] = doc
    return [doc_map[c] for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def chroma_filtrado(chroma_store, query: str, universidade: str) -> List[Document]:
    try:
        return chroma_store.similarity_search(query, k=TOP_K_EACH, filter={"universidade": universidade})
    except Exception:
        return []


def retrieval_hibrido(chroma_store, bm25, query: str) -> tuple:
    universidades        = extrair_universidades_da_query(query)
    tipo, top_k_dinamico = classificar_query(universidades)
    res_bm25   = bm25.invoke(query)
    res_global = chroma_store.similarity_search(query, k=TOP_K_EACH)
    res_filtrados = []
    for uni in universidades:
        res = chroma_filtrado(chroma_store, query, uni)
        res_filtrados.append(res if res else res_global)
    if not res_filtrados:
        res_filtrados = [res_global]
    candidatos = rrf_multi([res_bm25, res_global] + res_filtrados)
    return candidatos, top_k_dinamico, tipo


def rerankar(reranker, query: str, candidatos: List[Document], top_k: int) -> List[Document]:
    if not candidatos:
        return []
    scores    = reranker.predict([(query, doc.page_content) for doc in candidatos])
    ordenados = sorted(zip(scores, candidatos), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ordenados[:top_k]]


def formatar_contexto(chunks: List[Document]) -> str:
    return "\n\n".join(
        f"[Trecho {i+1}] {c.metadata['universidade']} (Pag. {c.metadata['pagina']}):\n{c.page_content}"
        for i, c in enumerate(chunks)
    )


def responder(chroma_store, bm25, reranker, llm, pergunta: str) -> dict:
    candidatos, top_k, tipo = retrieval_hibrido(chroma_store, bm25, pergunta)
    chunks_relevantes = rerankar(reranker, pergunta, candidatos, top_k)
    contexto          = formatar_contexto(chunks_relevantes)

    user_prompt = (
        "Com base nos seguintes trechos de regulamentacoes universitarias sobre IA:\n\n"
        + contexto
        + f"\n\nPergunta: {pergunta}"
    )

    resposta = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    fontes = sorted(set(
        f"{c.metadata['universidade']} (pag. {c.metadata['pagina']})"
        for c in chunks_relevantes
    ))

    return {"resposta": resposta.content, "fontes": fontes, "tipo": tipo, "top_k": top_k}

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Reg. IA — Universidades Federais",
    page_icon="⚖️",
    layout="wide",
)

# Injetar CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session state
if "historico" not in st.session_state:
    st.session_state["historico"] = []
if "pergunta_input" not in st.session_state:
    st.session_state["pergunta_input"] = ""
if "disparar_pergunta" not in st.session_state:
    st.session_state["disparar_pergunta"] = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
        <div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1rem;
                    color:#e8eaf0;letter-spacing:-0.02em;margin-bottom:0.25rem;">
            ⚖️ RegIA
        </div>
        <div style="font-family:'Sora',sans-serif;font-size:0.75rem;color:#8b93a8;line-height:1.5;">
            Consulta às regulamentações de uso de IA em universidades federais brasileiras,
            baseada nos documentos oficiais de cada instituição.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Carregamento do pipeline
    with st.spinner("Indexando documentos..."):
        chroma_store, bm25, reranker_model, llm, contagem, n_pdfs = construir_pipeline()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDFs", n_pdfs)
    with col2:
        st.metric("Chunks", sum(contagem.values()))

    st.divider()

    # Universidades clicáveis
    st.markdown('<span class="sidebar-label">Universidades</span>', unsafe_allow_html=True)
    unis_sorted = sorted(contagem.keys())

    for uni in unis_sorted:
        label = f"{uni}"
        if st.button(label, key=f"uni_{uni}"):
            st.session_state["pergunta_input"] = f"O que a {uni} regulamenta sobre uso de IA?"
            st.session_state["disparar_pergunta"] = True
            st.rerun()

    st.divider()

    # Exemplos de perguntas clicáveis
    st.markdown('<span class="sidebar-label">Exemplos de perguntas</span>', unsafe_allow_html=True)

    for i, exemplo in enumerate(EXEMPLOS_PERGUNTAS):
        if st.button(exemplo, key=f"ex_{i}"):
            st.session_state["pergunta_input"] = exemplo
            st.session_state["disparar_pergunta"] = True
            st.rerun()

# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>Regulamentações de IA</h1>
    <div class="subtitle">
        Políticas de uso de Inteligência Artificial nas universidades federais brasileiras —
        respostas baseadas exclusivamente nos documentos oficiais.
    </div>
</div>
""", unsafe_allow_html=True)

# Histórico de mensagens
for msg in st.session_state["historico"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("fontes"):
            with st.expander("📄 Fontes consultadas"):
                for fonte in msg["fontes"]:
                    st.markdown(
                        f'<span style="font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:0.78rem;color:#8b93a8;">→ {fonte}</span>',
                        unsafe_allow_html=True,
                    )

# Input do chat
pergunta = st.chat_input("Faça uma pergunta sobre as regulamentações de IA...")

# Disparar pergunta via clique na sidebar
if st.session_state.get("disparar_pergunta") and st.session_state.get("pergunta_input"):
    pergunta = st.session_state["pergunta_input"]
    st.session_state["pergunta_input"] = ""
    st.session_state["disparar_pergunta"] = False

if pergunta:
    st.session_state["historico"].append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            resultado = responder(chroma_store, bm25, reranker_model, llm, pergunta)

        st.markdown(resultado["resposta"])

        if resultado["fontes"]:
            with st.expander("📄 Fontes consultadas"):
                for fonte in resultado["fontes"]:
                    st.markdown(
                        f'<span style="font-family:\'JetBrains Mono\',monospace;'
                        f'font-size:0.78rem;color:#8b93a8;">→ {fonte}</span>',
                        unsafe_allow_html=True,
                    )

        tipo_label = {
            "especifica":  "específica",
            "comparativa": "comparativa",
            "geral":       "geral",
        }.get(resultado["tipo"], resultado["tipo"])

        st.markdown(
            f'<div style="margin-top:0.5rem;">'
            f'<span class="tipo-badge">✦ {tipo_label}</span>'
            f'<span class="chunks-badge">{resultado["top_k"]} chunks</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.session_state["historico"].append({
        "role":    "assistant",
        "content": resultado["resposta"],
        "fontes":  resultado["fontes"],
    })
