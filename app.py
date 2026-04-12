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
# CSS — estética escura RegIA
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Sora:wght@300;400;500&display=swap');

:root {
    --bg:          #0F1612;
    --bg2:         #141D17;
    --bg3:         #1A2620;
    --surface:     #1F2E24;
    --surface2:    #27382C;
    --verde:       #5DBE85;
    --verde-dim:   #3A8A5C;
    --verde-faint: rgba(93,190,133,0.09);
    --rosa:        #E8899A;
    --rosa-dim:    #C45E72;
    --texto:       #D9E8DC;
    --texto2:      #8AAF94;
    --texto3:      #4E7058;
    --borda:       rgba(93,190,133,0.14);
    --borda2:      rgba(93,190,133,0.28);
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--bg) !important;
    color: var(--texto) !important;
}

#MainMenu, footer, [data-testid="stToolbar"] {
    display: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--borda) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: var(--bg2) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
    color: var(--texto2) !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--texto) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--borda) !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--borda) !important;
    border-radius: 9px !important;
    padding: 9px 12px !important;
}
[data-testid="stMetricValue"] {
    color: var(--verde) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--texto3) !important;
    font-size: 0.68rem !important;
}

/* ── Título principal ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 600 !important;
    color: var(--texto) !important;
    font-size: 1.9rem !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--texto3) !important;
    font-size: 0.82rem !important;
    line-height: 1.6;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.5rem;
    border: 1px solid transparent !important;
    background: transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--bg3) !important;
    border-color: var(--borda) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: var(--surface) !important;
    border-color: rgba(232,137,154,0.18) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--surface2) !important;
    border: 1.5px solid var(--verde-dim) !important;
    border-radius: 50% !important;
}
[data-testid="chatAvatarIcon-user"] {
    background-color: var(--surface2) !important;
    border: 1.5px solid var(--rosa-dim) !important;
    border-radius: 50% !important;
}
[data-testid="stChatMessage"] p {
    font-size: 0.88rem !important;
    line-height: 1.75 !important;
    color: var(--texto) !important;
}
[data-testid="stChatMessage"] strong {
    color: var(--verde) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1.5px solid var(--borda2) !important;
    background-color: var(--bg2) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--verde-dim) !important;
    box-shadow: 0 0 0 3px rgba(93,190,133,0.07) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--texto) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--texto3) !important;
}
[data-testid="stChatInputSubmitButton"] {
    background-color: var(--verde) !important;
    border-radius: 9px !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background-color: var(--verde-dim) !important;
}

/* ── Expander (fontes) ── */
[data-testid="stExpander"] {
    border: 1px solid var(--borda) !important;
    border-radius: 9px !important;
    background-color: var(--bg2) !important;
    margin-top: 0.5rem;
}
[data-testid="stExpander"] summary {
    color: var(--verde) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] p {
    font-size: 0.78rem !important;
    color: var(--texto2) !important;
}

[data-testid="stSpinner"] { color: var(--verde) !important; }

hr { border-color: var(--borda) !important; }

[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left: 4px solid var(--rosa) !important;
    background: rgba(232,137,154,0.09) !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 10px; }
* { scrollbar-width: thin; scrollbar-color: var(--surface2) transparent; }

@media (max-width: 768px) {
    h1 { font-size: 1.4rem !important; }
    [data-testid="stChatMessage"] { padding: 0.75rem 1rem !important; }
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
# Pipeline
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
        todas_paginas.extend(extrair_texto_pdf(pdf))

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
    chunks_relevantes       = rerankar(reranker, pergunta, candidatos, top_k)
    contexto                = formatar_contexto(chunks_relevantes)

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

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Carregamento ──
with st.spinner("🌱 Carregando modelos e indexando documentos..."):
    chroma_store, bm25, reranker_model, llm, contagem, n_pdfs = construir_pipeline()

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        """
        <div style="padding:24px 16px 20px;border-bottom:1px solid rgba(93,190,133,0.14);margin-bottom:16px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
            <div style="width:32px;height:32px;background:#C45E72;border-radius:50% 50% 50% 6px;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
                <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 0 0 8 20
                         C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/>
              </svg>
            </div>
            <span style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:600;
                         color:#E8EDE9;letter-spacing:0.02em;line-height:1">RegIA</span>
          </div>
          <div style="font-size:0.68rem;color:#4E7058;letter-spacing:0.08em;text-transform:uppercase">
            Regulamentações · Universidades Federais
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Documentos indexados")
    col1, col2 = st.columns(2)
    col1.metric("PDFs", n_pdfs)
    col2.metric("Chunks", sum(contagem.values()))

    st.divider()
    st.subheader("Universidades disponíveis")
    for uni in sorted(contagem.keys()):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:5px 4px;"
            f"font-size:0.8rem;color:#8AAF94;'>"
            f"<span style='width:5px;height:5px;border-radius:50%;background:#E8899A;"
            f"display:inline-block;flex-shrink:0'></span>{uni}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Exemplos de perguntas")
    st.markdown(
        "<div style='font-size:0.8rem;line-height:1.9;color:#8AAF94;'>"
        "· A UFPB permite uso de IA em trabalhos acadêmicos?<br>"
        "· Compare as políticas da UFMG e da UFRJ sobre IA em provas.<br>"
        "· Quais universidades proíbem completamente o uso de IA?<br>"
        "· O que a UNIFESP diz sobre plágio com IA?"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Logo e título ──
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown(
        """
        <div style="width:48px;height:48px;background:#C45E72;
                    border-radius:50% 50% 50% 10px;
                    display:flex;align-items:center;justify-content:center;margin-top:6px">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
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

# ── Histórico ──
if "historico" not in st.session_state:
    st.session_state["historico"] = []

for msg in st.session_state["historico"]:
    with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("fontes"):
            with st.expander("📄 Fontes consultadas"):
                for fonte in msg["fontes"]:
                    st.text(fonte)

# ── Entrada ──
pergunta = st.chat_input("Digite sua pergunta sobre as regulamentações de IA...")

if pergunta:
    st.session_state["historico"].append({"role": "user", "content": pergunta})

    with st.chat_message("user", avatar="👤"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="🌿"):
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

        st.caption(f"🌿 {tipo_label} · {resultado['top_k']} chunks no contexto")

    st.session_state["historico"].append({
        "role":    "assistant",
        "content": resultado["resposta"],
        "fontes":  resultado["fontes"],
    })
