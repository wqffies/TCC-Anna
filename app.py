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

LLM_MODEL      = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 400
TOP_K_EACH    = 10

TOP_K_ESPECIFICA  = 5
TOP_K_COMPARATIVA = 8
TOP_K_GERAL       = 12

ALIASES_UNIVERSIDADE = {
    "Unifesp": "UNIFESP", "unifesp":"UNIFESP", "sao paulo": "UNIFESP", "são paulo": "UNIFESP",
    "ufpb": "UFPB", "paraiba": "UFPB", "paraíba": "UFPB",
    "ufc": "UFC", "ceara": "UFC", "ceára": "UFC",
    "ufba": "UFBA", "bahia": "UFBA",
    "ufdpar": "UFDPAR", "delta do parnaiba": "UFDPAR",
    "uff": "UFF", "fluminense": "UFF",
    "ufg": "UFG", "goias": "UFG", "goiás": "UFG",
    "ufma": "UFMA", "maranhao": "UFMA", "maranhão": "UFMA",
    "ufmg": "UFMG", "minas gerais": "UFMG",
    "ufms": "UFMS", "mato grosso" : "UFMS",
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
            "Verifique se os arquivos estao no repositorio."
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
    page_title="Regulamentacoes de IA - Universidades Federais",
    layout="wide",
)

# --- CSS: esconde barra superior do Streamlit e estiliza botões da sidebar ---
st.markdown("""
<style>
/* Esconde apenas os botões da toolbar (deploy, fork, share, menu),
   mas mantém o botão de abrir/fechar a sidebar */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
#MainMenu,
header[data-testid="stHeader"] .stAppToolbar,
header[data-testid="stHeader"] .stToolbar,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}
/* Compensa o espaço que a barra ocupava */
.block-container {
    padding-top: 2rem !important;
}

/* Botões de pergunta sugerida */
.sidebar-btn button {
    width: 100% !important;
    text-align: left !important;
    background-color: transparent !important;
    border: 1px solid rgba(128,128,128,0.3) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.6rem !important;
    font-size: 0.82rem !important;
    color: inherit !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.sidebar-btn button:hover {
    background-color: rgba(128,128,128,0.12) !important;
    border-color: rgba(128,128,128,0.6) !important;
}

/* Botões de universidade */
.uni-btn button {
    width: 100% !important;
    text-align: left !important;
    background-color: transparent !important;
    border: 1px solid rgba(100,149,237,0.35) !important;
    border-radius: 6px !important;
    padding: 0.3rem 0.6rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: inherit !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
    letter-spacing: 0.03em !important;
}
.uni-btn button:hover {
    background-color: rgba(100,149,237,0.12) !important;
    border-color: rgba(100,149,237,0.7) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Regulamentações de IA em Universidades Federais")
st.caption(
    "Consulte as políticas de uso de Inteligência Artificial das universidades federais brasileiras. "
    "As respostas são baseadas exclusivamente nos documentos oficiais de cada instituição."
)

with st.spinner("Carregando modelos e indexando documentos..."):
    chroma_store, bm25, reranker_model, llm, contagem, n_pdfs = construir_pipeline()

# ---------------------------------------------------------------------------
# Session state para pergunta pré-preenchida via sidebar
# ---------------------------------------------------------------------------
if "pergunta_sugerida" not in st.session_state:
    st.session_state["pergunta_sugerida"] = None
if "historico" not in st.session_state:
    st.session_state["historico"] = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sobre")
    st.write(
        "Sistema de consulta às regulamentações de uso de Inteligência Artificial "
        "em universidades federais brasileiras, baseado nos documentos oficiais de cada instituição."
    )

    st.divider()
    st.subheader("Documentos indexados")
    st.metric("PDFs carregados", n_pdfs)
    st.metric("Total de chunks", sum(contagem.values()))

    st.divider()
    st.subheader("Universidades disponíveis")
    for uni in sorted(contagem.keys()):
        st.markdown('<div class="uni-btn">', unsafe_allow_html=True)
        if st.button(uni, key=f"uni_{uni}"):
            st.session_state["pergunta_sugerida"] = f"O que a {uni} diz sobre o uso de IA?"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Exemplos de perguntas")
    perguntas_exemplo = [
        "A UFPB permite uso de IA em trabalhos acadêmicos?",
        "Compare as políticas da UFMG e da UFRJ sobre IA em provas.",
        "Quais universidades proíbem completamente o uso de IA?",
        "O que a UNIFESP diz sobre plágio com IA?",
    ]
    for pergunta_ex in perguntas_exemplo:
        st.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
        if st.button(pergunta_ex, key=f"ex_{pergunta_ex}"):
            st.session_state["pergunta_sugerida"] = pergunta_ex
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Historico do chat
# ---------------------------------------------------------------------------
for msg in st.session_state["historico"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("fontes"):
            with st.expander("Fontes consultadas"):
                for fonte in msg["fontes"]:
                    st.text(fonte)

# ---------------------------------------------------------------------------
# Entrada — suporte a pergunta sugerida via sidebar
# ---------------------------------------------------------------------------
sugerida = st.session_state["pergunta_sugerida"]
if sugerida:
    st.session_state["pergunta_sugerida"] = None

pergunta = st.chat_input("Digite sua pergunta sobre as regulamentações de IA...") or sugerida

if pergunta:
    st.session_state["historico"].append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            resultado = responder(chroma_store, bm25, reranker_model, llm, pergunta)

        st.markdown(resultado["resposta"])

        if resultado["fontes"]:
            with st.expander("Fontes consultadas"):
                for fonte in resultado["fontes"]:
                    st.text(fonte)

        tipo_label = {
            "especifica":  "consulta específica",
            "comparativa": "consulta comparativa",
            "geral":       "consulta geral",
        }.get(resultado["tipo"], resultado["tipo"])

        st.caption(f"Tipo: {tipo_label} | Chunks no contexto: {resultado['top_k']}")

    st.session_state["historico"].append({
        "role":    "assistant",
        "content": resultado["resposta"],
        "fontes":  resultado["fontes"],
    })
