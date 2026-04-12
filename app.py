"""
RegIA - Regulamentacoes de IA em Universidades Federais
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
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 400
TOP_K_EACH    = 10

TOP_K_ESPECIFICA  = 5
TOP_K_COMPARATIVA = 8
TOP_K_GERAL       = 12

ALIASES_UNIVERSIDADE = {
    "ufpb":           "UFPB",
    "ufc":            "UFC",
    "ufac":           "UFAC",
    "ufba":           "UFBA",
    "ufdpar":         "UFDPAR",
    "ufersa":         "UFERSA",
    "uff":            "UFF",
    "ufg":            "UFG",
    "ufma":           "UFMA",
    "ufmg":           "UFMG",
    "ufms":           "UFMS",
    "ufop":           "UFOP",
    "ufrj":           "UFRJ",
    "ufu":            "UFU",
    "unifal":         "UNIFAL",
    "unifesp":        "UNIFESP",
    "unir":           "UNIR",
    "paraiba":        "UFPB",
    "paraíba":        "UFPB",
    "ceara":          "UFC",
    "ceará":          "UFC",
    "acre":           "UFAC",
    "bahia":          "UFBA",
    "fluminense":     "UFF",
    "goias":          "UFG",
    "goiás":          "UFG",
    "maranhao":       "UFMA",
    "maranhão":       "UFMA",
    "minas gerais":   "UFMG",
    "ouro preto":     "UFOP",
    "rio de janeiro": "UFRJ",
    "uberlandia":     "UFU",
    "uberlândia":     "UFU",
    "alfenas":        "UNIFAL",
    "sao paulo":      "UNIFESP",
    "são paulo":      "UNIFESP",
    "rondonia":       "UNIR",
    "rondônia":       "UNIR",
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

# ---------------------------------------------------------------------------
# CSS customizado - tema botanico RegIA
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Variaveis de cor ── */
:root {
    --rosa:        #D97B9A;
    --rosa-claro:  #F5D0DE;
    --rosa-escuro: #8C3A56;
    --rosa-fundo:  #FDF0F4;
    --verde:       #4A7C59;
    --verde-claro: #B8D9C4;
    --verde-escuro:#2C4E38;
    --verde-fundo: #EEF5F0;
    --creme:       #FAF7F2;
    --texto:       #2C2C2A;
    --texto-muted: #7A7570;
    --borda:       rgba(74,124,89,0.18);
}

/* ── Reset e base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Fundo geral */
.stApp {
    background-color: var(--creme) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--verde-escuro) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: var(--verde-escuro) !important;
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] * {
    color: var(--verde-claro) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #fff !important;
    font-family: 'Cormorant Garamond', serif !important;
}
[data-testid="stSidebar"] .stMetric label {
    color: var(--verde-claro) !important;
    font-size: 0.7rem !important;
    opacity: 0.75;
}
[data-testid="stSidebar"] .stMetric [data-testid="metric-container"] > div:last-child {
    color: var(--rosa-claro) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.8rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    font-size: 0.8rem !important;
    opacity: 0.8;
    line-height: 1.6;
}
[data-testid="stSidebar"] strong {
    color: var(--rosa-claro) !important;
    opacity: 1 !important;
}

/* ── Titulo principal ── */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    color: var(--verde-escuro) !important;
    font-size: 2rem !important;
    letter-spacing: 0.01em;
}

/* Caption / subtitulo */
[data-testid="stCaptionContainer"] p {
    color: var(--texto-muted) !important;
    font-size: 0.82rem !important;
    line-height: 1.6;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.5rem;
    border: 1px solid transparent !important;
}

/* Mensagem do assistente */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--verde-fundo) !important;
    border-color: var(--verde-claro) !important;
}

/* Mensagem do usuario */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: var(--rosa-fundo) !important;
    border-color: var(--rosa-claro) !important;
}

/* Avatares */
[data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--verde) !important;
    color: #fff !important;
    border-radius: 50% !important;
}
[data-testid="chatAvatarIcon-user"] {
    background-color: var(--rosa) !important;
    color: #fff !important;
    border-radius: 50% !important;
}

/* Texto das mensagens */
[data-testid="stChatMessage"] p {
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    color: var(--texto) !important;
}
[data-testid="stChatMessage"] strong {
    color: var(--verde-escuro) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 28px !important;
    border: 1.5px solid var(--borda) !important;
    background-color: #fff !important;
    padding: 0.25rem 0.5rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--verde) !important;
    box-shadow: 0 0 0 2px rgba(74,124,89,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--texto) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--texto-muted) !important;
}
[data-testid="stChatInputSubmitButton"] {
    background-color: var(--verde) !important;
    border-radius: 50% !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background-color: var(--verde-escuro) !important;
}

/* ── Expander (fontes) ── */
[data-testid="stExpander"] {
    border: 1px solid var(--verde-claro) !important;
    border-radius: 10px !important;
    background-color: #fff !important;
    margin-top: 0.5rem;
}
[data-testid="stExpander"] summary {
    color: var(--verde) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] p {
    font-size: 0.78rem !important;
    color: var(--texto-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Caption de tipo de consulta ── */
[data-testid="stCaptionContainer"] {
    margin-top: 0.25rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--verde) !important;
}

/* ── Metric cards na sidebar ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.8rem !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Divider ── */
hr {
    border-color: var(--borda) !important;
}

/* ── Botao de erro ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left: 4px solid var(--rosa) !important;
    background: var(--rosa-fundo) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--verde-claro);
    border-radius: 10px;
}

/* ── Mobile responsivo ── */
@media (max-width: 768px) {
    h1 { font-size: 1.5rem !important; }
    [data-testid="stChatMessage"] {
        padding: 0.8rem 1rem !important;
    }
}
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

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        groq_api_key=groq_api_key,
    )

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        st.error(
            f"Nenhum PDF encontrado na pasta '{PDF_DIR.name}/'. "
            "Verifique se os arquivos estão no repositório."
        )
        st.stop()

    todas_paginas = []
    for pdf in pdfs:
        paginas = extrair_texto_pdf(pdf)
        todas_paginas.extend(paginas)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documentos = []
    for pagina in todas_paginas:
        documentos.append(Document(
            page_content=f"Regulamentacao de IA da {pagina['universidade']}: {pagina['texto']}",
            metadata={
                "arquivo":      pagina["arquivo"],
                "universidade": pagina["universidade"],
                "pagina":       pagina["pagina"],
            },
        ))

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
        return chroma_store.similarity_search(
            query, k=TOP_K_EACH, filter={"universidade": universidade}
        )
    except Exception:
        return []


def retrieval_hibrido(chroma_store, bm25, query: str) -> tuple:
    universidades         = extrair_universidades_da_query(query)
    tipo, top_k_dinamico  = classificar_query(universidades)

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
        f"{c.metadata['universidade']} (pág. {c.metadata['pagina']})"
        for c in chunks_relevantes
    ))

    return {"resposta": resposta.content, "fontes": fontes, "tipo": tipo, "top_k": top_k}

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RegIA · Regulamentações de IA",
    page_icon="🌿",
    layout="wide",
)

# Injeta CSS personalizado
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Logo e titulo ──
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown(
        """
        <div style="
            width:48px; height:48px;
            background:#D97B9A;
            border-radius:50% 50% 50% 10px;
            display:flex; align-items:center; justify-content:center;
            margin-top:4px;
        ">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
                <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 0 0 8 20
                         C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_title:
    st.title("RegIA")

st.caption(
    "Consulte as políticas de uso de Inteligência Artificial das universidades federais brasileiras. "
    "As respostas são baseadas exclusivamente nos documentos oficiais de cada instituição."
)

# ── Carregamento ──
with st.spinner("🌱 Carregando modelos e indexando documentos..."):
    chroma_store, bm25, reranker_model, llm, contagem, n_pdfs = construir_pipeline()

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        """
        <div style="
            font-family:'Cormorant Garamond',serif;
            font-size:1.6rem;
            font-weight:600;
            color:#fff;
            display:flex; align-items:center; gap:10px;
            margin-bottom:4px;
        ">
            <span style="
                width:32px; height:32px;
                background:#D97B9A;
                border-radius:50% 50% 50% 6px;
                display:inline-flex; align-items:center; justify-content:center;
            ">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
                    <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 0 0 8 20
                             C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/>
                </svg>
            </span>
            RegIA
        </div>
        <p style="font-size:0.72rem; color:#B8D9C4; opacity:0.7; margin:0 0 1rem;">
            Regulamentos de IA · Universidades Federais
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Documentos indexados")
    col1, col2 = st.columns(2)
    col1.metric("PDFs", n_pdfs)
    col2.metric("Chunks", sum(contagem.values()))

    st.divider()
    st.subheader("Universidades disponíveis")
    for uni in sorted(contagem.keys()):
        st.markdown(
            f"<span style='display:inline-flex;align-items:center;gap:6px;font-size:0.8rem;'>"
            f"<span style='width:6px;height:6px;border-radius:50%;background:#D97B9A;display:inline-block;'></span>"
            f"{uni}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Exemplos de perguntas")
    st.markdown(
        "- A UFPB permite uso de IA em trabalhos acadêmicos?\n"
        "- Compare as políticas da UFMG e da UFRJ sobre IA em provas.\n"
        "- Quais universidades proíbem completamente o uso de IA?\n"
        "- O que a UNIFESP diz sobre plágio com IA?"
    )

# ── Historico ──
if "historico" not in st.session_state:
    st.session_state["historico"] = []

for msg in st.session_state["historico"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("fontes"):
            with st.expander("📄 Fontes consultadas"):
                for fonte in msg["fontes"]:
                    st.text(fonte)

# ── Entrada ──
pergunta = st.chat_input("Digite sua pergunta sobre as regulamentações de IA...")

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
                    st.text(fonte)

        tipo_label = {
            "especifica":  "consulta específica",
            "comparativa": "consulta comparativa",
            "geral":       "consulta geral",
        }.get(resultado["tipo"], resultado["tipo"])

        st.caption(
            f"🌿 {tipo_label} · {resultado['top_k']} chunks no contexto"
        )

    st.session_state["historico"].append({
        "role":    "assistant",
        "content": resultado["resposta"],
        "fontes":  resultado["fontes"],
    })
