"""
RegIA - Regulamentacoes de IA em Universidades Federais
Interface nativa do Streamlit com CSS customizado (sem iframe/bridge JS)
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
    "ufpb": "UFPB", "ufc": "UFC", "ufac": "UFAC", "ufba": "UFBA",
    "ufdpar": "UFDPAR", "ufersa": "UFERSA", "uff": "UFF", "ufg": "UFG",
    "ufma": "UFMA", "ufmg": "UFMG", "ufms": "UFMS", "ufop": "UFOP",
    "ufrj": "UFRJ", "ufu": "UFU", "unifal": "UNIFAL", "unifesp": "UNIFESP",
    "unir": "UNIR", "paraiba": "UFPB", "paraíba": "UFPB", "ceara": "UFC",
    "ceará": "UFC", "acre": "UFAC", "bahia": "UFBA", "fluminense": "UFF",
    "goias": "UFG", "goiás": "UFG", "maranhao": "UFMA", "maranhão": "UFMA",
    "minas gerais": "UFMG", "ouro preto": "UFOP", "rio de janeiro": "UFRJ",
    "uberlandia": "UFU", "uberlândia": "UFU", "alfenas": "UNIFAL",
    "sao paulo": "UNIFESP", "são paulo": "UNIFESP", "rondonia": "UNIR",
    "rondônia": "UNIR",
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
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, groq_api_key=groq_api_key)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        st.error(f"Nenhum PDF encontrado na pasta '{PDF_DIR.name}/'.")
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
        return chroma_store.similarity_search(
            query, k=TOP_K_EACH, filter={"universidade": universidade}
        )
    except Exception:
        return []


def retrieval_hibrido(chroma_store, bm25, query: str) -> tuple:
    universidades        = extrair_universidades_da_query(query)
    tipo, top_k_dinamico = classificar_query(universidades)
    res_bm25             = bm25.invoke(query)
    res_global           = chroma_store.similarity_search(query, k=TOP_K_EACH)
    res_filtrados        = []
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
        f"{c.metadata['universidade']} (pag. {c.metadata['pagina']})"
        for c in chunks_relevantes
    ))
    return {"resposta": resposta.content, "fontes": fontes, "tipo": tipo, "top_k": top_k}

# ---------------------------------------------------------------------------
# Streamlit — interface nativa com CSS do design original
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RegIA · Regulamentações de IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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

/* App background */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--bg) !important;
  font-family: 'Sora', sans-serif !important;
  color: var(--texto) !important;
}

/* Remove chrome nativo */
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--borda) !important;
}
[data-testid="stSidebar"] * {
  color: var(--texto2) !important;
  font-family: 'Sora', sans-serif !important;
}

/* Chat messages container */
[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 4px 0 !important;
}

/* Bubble assistente */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
  background: var(--bg3) !important;
  border: 1px solid var(--borda) !important;
  border-radius: 14px !important;
  border-top-left-radius: 3px !important;
  padding: 13px 17px !important;
  color: #D9E8DC !important;
  font-size: 13.5px !important;
  line-height: 1.75 !important;
}

/* Bubble usuário */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
  background: var(--surface) !important;
  border: 1px solid rgba(232,137,154,0.18) !important;
  border-radius: 14px !important;
  border-top-right-radius: 3px !important;
  padding: 13px 17px !important;
  color: #D9E8DC !important;
  font-size: 13.5px !important;
  line-height: 1.75 !important;
}

/* Avatares */
[data-testid="chatAvatarIcon-assistant"] {
  background: var(--surface2) !important;
  border: 1.5px solid var(--verde-dim) !important;
  color: var(--verde) !important;
  border-radius: 50% !important;
}
[data-testid="chatAvatarIcon-user"] {
  background: var(--surface2) !important;
  border: 1.5px solid var(--rosa-dim) !important;
  color: var(--rosa) !important;
  border-radius: 50% !important;
}

/* Chat input */
[data-testid="stChatInput"] {
  background: var(--bg2) !important;
  border: 1.5px solid var(--borda2) !important;
  border-radius: 14px !important;
  margin: 0 28px 20px !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--verde-dim) !important;
  box-shadow: 0 0 0 3px rgba(93,190,133,0.07) !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: #D9E8DC !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 13.5px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--texto3) !important; }
[data-testid="stChatInput"] button { background: var(--verde) !important; border-radius: 9px !important; }

/* Métricas na sidebar */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--borda) !important;
  border-radius: 9px !important;
  padding: 9px 12px !important;
}
[data-testid="stMetricValue"] {
  color: var(--verde) !important;
  font-family: 'Playfair Display', serif !important;
  font-size: 22px !important;
}
[data-testid="stMetricLabel"] { color: var(--texto3) !important; font-size: 10px !important; }

/* Expander de fontes */
[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--borda) !important;
  border-radius: 9px !important;
  margin-top: 4px !important;
}
[data-testid="stExpander"] summary { color: var(--verde) !important; font-size: 10.5px !important; }

/* Botões de exemplo */
.stButton > button {
  background: var(--surface) !important;
  border: 1px solid var(--borda) !important;
  border-radius: 8px !important;
  color: var(--texto2) !important;
  font-size: 11.5px !important;
  font-family: 'Sora', sans-serif !important;
  width: 100% !important;
  text-align: left !important;
  padding: 8px 11px !important;
}
.stButton > button:hover {
  background: var(--verde-faint) !important;
  border-color: var(--verde-dim) !important;
  color: var(--verde) !important;
}

/* Container do chat com scroll */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: transparent !important;
  border: none !important;
}

/* Scrollbar */
* { scrollbar-width: thin; scrollbar-color: var(--surface2) transparent; }

/* Tag de consulta */
.consulta-tag {
  display: inline-block;
  font-size: 10.5px;
  color: var(--texto2);
  padding: 2px 9px;
  border: 1px solid var(--borda);
  border-radius: 20px;
  background: var(--bg2);
  margin-top: 6px;
  font-family: 'Sora', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── Carrega pipeline ────────────────────────────────────────────────────────
with st.spinner("🌱 Carregando modelos e indexando documentos..."):
    chroma_store, bm25, reranker_model, llm, contagem, n_pdfs = construir_pipeline()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:26px 22px 20px;border-bottom:1px solid var(--borda);margin-bottom:18px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
        <div style="width:36px;height:36px;background:#C45E72;border-radius:50% 50% 50% 9px;
                    display:flex;align-items:center;justify-content:center">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="#fff">
            <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 008 20C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/>
          </svg>
        </div>
        <span style="font-family:'Playfair Display',serif;font-size:24px;font-weight:600;
                     color:#E8EDE9;letter-spacing:0.02em;line-height:1">RegIA</span>
      </div>
      <div style="font-size:10.5px;color:var(--texto3);letter-spacing:0.07em;text-transform:uppercase">
        Regulamentações · Universidades Federais
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:9.5px;letter-spacing:0.13em;text-transform:uppercase;color:var(--texto3);margin-bottom:10px;padding:0 4px">Base de dados</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PDFs", n_pdfs)
    with col2:
        total = sum(contagem.values())
        st.metric("Chunks", f"{total/1000:.1f}k" if total > 999 else total)

    st.markdown("<hr style='border:none;border-top:1px solid var(--borda);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:9.5px;letter-spacing:0.13em;text-transform:uppercase;color:var(--texto3);margin-bottom:10px;padding:0 4px">Universidades disponíveis</div>', unsafe_allow_html=True)

    for uni in sorted(contagem.keys()):
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:6px 9px;
                        border-radius:7px;font-size:12px;color:var(--texto2)">
          <span style="width:5px;height:5px;border-radius:50%;background:#E8899A;display:inline-block;flex-shrink:0"></span>
          {uni}</div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid var(--borda);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:9.5px;letter-spacing:0.13em;text-transform:uppercase;color:var(--texto3);margin-bottom:10px;padding:0 4px">Exemplos de perguntas</div>', unsafe_allow_html=True)

    exemplos = [
        "A UFPB permite uso de IA em trabalhos acadêmicos?",
        "Compare as políticas da UFMG e da UFRJ sobre IA em provas.",
        "Quais universidades proíbem completamente o uso de IA?",
        "O que a UNIFESP diz sobre plágio com IA?",
    ]
    for ex in exemplos:
        if st.button(ex, key=f"ex_{ex[:30]}"):
            st.session_state["exemplo_selecionado"] = ex
            st.rerun()

# ── Topbar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:18px 28px 16px;border-bottom:1px solid var(--borda);
            display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
  <span style="font-family:'Playfair Display',serif;font-size:17px;color:#E8EDE9">
    Consulta de Regulamentações
  </span>
  <span style="font-size:11.5px;color:var(--texto3)">
    Respostas baseadas exclusivamente nos documentos oficiais
  </span>
</div>
""", unsafe_allow_html=True)

# ── Histórico ────────────────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state["historico"] = []

# Área de chat
chat_area = st.container(height=520, border=False)

with chat_area:
    if not st.session_state["historico"]:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    text-align:center;padding:60px 20px">
          <div style="width:60px;height:60px;background:#C45E72;border-radius:50% 50% 50% 13px;
                      display:flex;align-items:center;justify-content:center;margin:0 auto 18px">
            <svg viewBox="0 0 24 24" width="30" height="30" fill="#fff">
              <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 008 20C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/>
            </svg>
          </div>
          <h2 style="font-family:'Playfair Display',serif;font-size:26px;font-weight:600;
                     color:#E8EDE9;margin-bottom:10px">Olá, sou o RegIA</h2>
          <p style="font-size:13.5px;color:#8AAF94;max-width:360px;line-height:1.7">
            Seu assistente para consultar as regulamentações de uso de Inteligência Artificial
            nas universidades federais brasileiras.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state["historico"]:
            with st.chat_message(msg["role"], avatar="RI" if msg["role"] == "assistant" else "VC"):
                st.write(msg["texto"])
                if msg["role"] == "assistant" and msg.get("fontes"):
                    labels = {
                        "especifica":  "consulta específica",
                        "comparativa": "consulta comparativa",
                        "geral":       "consulta geral",
                    }
                    tipo_label = labels.get(msg.get("tipo", ""), msg.get("tipo", ""))
                    st.markdown(
                        f'<span class="consulta-tag">{tipo_label} · {msg.get("top_k","")} chunks</span>',
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"📄 {len(msg['fontes'])} fonte(s)"):
                        for fonte in msg["fontes"]:
                            st.markdown(f"• {fonte}")

# ── Input ─────────────────────────────────────────────────────────────────────
exemplo = st.session_state.pop("exemplo_selecionado", None)
pergunta_digitada = st.chat_input(
    "Pergunte sobre regulamentações de IA nas universidades federais...",
)
pergunta = pergunta_digitada or exemplo

if pergunta:
    st.session_state["historico"].append({"role": "user", "texto": pergunta})
    with st.spinner("Consultando documentos..."):
        resultado = responder(chroma_store, bm25, reranker_model, llm, pergunta)
    st.session_state["historico"].append({
        "role":   "assistant",
        "texto":  resultado["resposta"],
        "fontes": resultado["fontes"],
        "tipo":   resultado["tipo"],
        "top_k":  resultado["top_k"],
    })
    st.rerun()
