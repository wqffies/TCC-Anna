"""
RegIA - Regulamentacoes de IA em Universidades Federais
Interface customizada via st.components.v1.html()
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List

import fitz
import streamlit as st
import streamlit.components.v1 as components
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
# HTML da interface completa
# ---------------------------------------------------------------------------

HTML_UI = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Sora:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:           #0F1612;
  --bg2:          #141D17;
  --bg3:          #1A2620;
  --surface:      #1F2E24;
  --surface2:     #27382C;
  --verde:        #5DBE85;
  --verde-dim:    #3A8A5C;
  --verde-faint:  rgba(93,190,133,0.09);
  --verde-glow:   rgba(93,190,133,0.15);
  --rosa:         #E8899A;
  --rosa-dim:     #C45E72;
  --rosa-faint:   rgba(232,137,154,0.09);
  --texto:        #D9E8DC;
  --texto2:       #8AAF94;
  --texto3:       #4E7058;
  --borda:        rgba(93,190,133,0.14);
  --borda2:       rgba(93,190,133,0.28);
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--texto);
  font-family: 'Sora', sans-serif;
  font-size: 14px;
  line-height: 1.65;
  overflow: hidden;
}

/* ─── Layout raiz ─── */
.shell {
  display: flex;
  height: 100vh;
  width: 100%;
}

/* ─── Sidebar ─── */
.sidebar {
  width: 252px;
  flex-shrink: 0;
  background: var(--bg2);
  border-right: 1px solid var(--borda);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.brand {
  padding: 26px 22px 20px;
  border-bottom: 1px solid var(--borda);
  flex-shrink: 0;
}

.brand-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
}

.leaf-icon {
  width: 36px;
  height: 36px;
  background: var(--rosa-dim);
  border-radius: 50% 50% 50% 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.leaf-icon svg { width: 18px; height: 18px; fill: #fff; }

.brand-name {
  font-family: 'Playfair Display', serif;
  font-size: 24px;
  font-weight: 600;
  color: #E8EDE9;
  letter-spacing: 0.02em;
  line-height: 1;
}

.brand-tagline {
  font-size: 10.5px;
  color: var(--texto3);
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.sidebar-body {
  flex: 1;
  overflow-y: auto;
  padding: 18px 0;
  scrollbar-width: thin;
  scrollbar-color: var(--surface2) transparent;
}

.sidebar-section { padding: 0 18px; margin-bottom: 22px; }

.sec-label {
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--texto3);
  margin-bottom: 10px;
}

.stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }

.stat-box {
  background: var(--surface);
  border: 1px solid var(--borda);
  border-radius: 9px;
  padding: 9px 12px;
}

.stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 22px;
  color: var(--verde);
  line-height: 1;
}

.stat-lbl { font-size: 10px; color: var(--texto3); margin-top: 2px; }

.divider {
  border: none;
  border-top: 1px solid var(--borda);
  margin: 2px 18px 18px;
}

.uni-list { display: flex; flex-direction: column; gap: 2px; }

.uni-tag {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 9px;
  border-radius: 7px;
  font-size: 12px;
  color: var(--texto2);
  cursor: pointer;
  transition: background 0.14s, color 0.14s;
  border: 1px solid transparent;
  user-select: none;
}

.uni-tag:hover {
  background: var(--verde-faint);
  border-color: var(--borda);
  color: var(--verde);
}

.uni-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--rosa);
  flex-shrink: 0;
}

.examples { display: flex; flex-direction: column; gap: 5px; }

.example-chip {
  background: var(--surface);
  border: 1px solid var(--borda);
  border-radius: 8px;
  padding: 8px 11px;
  font-size: 11.5px;
  color: var(--texto2);
  cursor: pointer;
  transition: background 0.14s, border-color 0.14s, color 0.14s;
  line-height: 1.45;
  user-select: none;
}

.example-chip:hover {
  background: var(--verde-faint);
  border-color: var(--verde-dim);
  color: var(--verde);
}

/* ─── Main ─── */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--bg);
}

.topbar {
  padding: 18px 28px 16px;
  border-bottom: 1px solid var(--borda);
  flex-shrink: 0;
  display: flex;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
}

.topbar-title {
  font-family: 'Playfair Display', serif;
  font-size: 17px;
  font-weight: 400;
  color: #E8EDE9;
}

.topbar-sub {
  font-size: 11.5px;
  color: var(--texto3);
}

/* ─── Área de chat ─── */
.chat-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 28px 28px 12px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  scrollbar-width: thin;
  scrollbar-color: var(--surface2) transparent;
}

/* Boas-vindas */
.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  text-align: center;
  padding: 40px 20px;
  animation: fadeUp 0.5s ease both;
}

.welcome-icon {
  width: 60px; height: 60px;
  background: var(--rosa-dim);
  border-radius: 50% 50% 50% 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 18px;
}

.welcome-icon svg { width: 30px; height: 30px; fill: #fff; }

.welcome h2 {
  font-family: 'Playfair Display', serif;
  font-size: 26px;
  font-weight: 600;
  color: #E8EDE9;
  margin-bottom: 10px;
}

.welcome p {
  font-size: 13.5px;
  color: var(--texto2);
  max-width: 360px;
  line-height: 1.7;
}

/* ─── Mensagens ─── */
.msg-row {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  animation: fadeUp 0.28s ease both;
}

.msg-row.user { flex-direction: row-reverse; }

.avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 500;
  flex-shrink: 0;
  letter-spacing: 0.02em;
}

.avatar.bot {
  background: var(--surface2);
  border: 1.5px solid var(--verde-dim);
  color: var(--verde);
}

.avatar.user {
  background: var(--surface2);
  border: 1.5px solid var(--rosa-dim);
  color: var(--rosa);
}

.bubble-wrap { max-width: 78%; display: flex; flex-direction: column; }
.msg-row.user .bubble-wrap { align-items: flex-end; }

.bubble {
  padding: 13px 17px;
  border-radius: 14px;
  font-size: 13.5px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
}

/* Texto do bot: fundo levemente diferente, borda verde sutil */
.bubble.bot {
  background: var(--bg3);
  border: 1px solid var(--borda);
  color: #D9E8DC;           /* contraste alto sobre bg3 */
  border-top-left-radius: 3px;
}

/* Texto do usuário: fundo levemente rosa */
.bubble.user {
  background: var(--surface);
  border: 1px solid rgba(232,137,154,0.18);
  color: #D9E8DC;           /* mesmo contraste */
  border-top-right-radius: 3px;
}

.meta {
  margin-top: 7px;
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
}

.meta-tag {
  font-size: 10.5px;
  color: var(--texto2);
  padding: 2px 9px;
  border: 1px solid var(--borda);
  border-radius: 20px;
  background: var(--bg2);
}

.fontes-toggle {
  font-size: 10.5px;
  color: var(--verde);
  background: var(--verde-faint);
  border: 1px solid rgba(93,190,133,0.22);
  border-radius: 20px;
  padding: 2px 10px;
  cursor: pointer;
  font-family: 'Sora', sans-serif;
  transition: background 0.14s;
}

.fontes-toggle:hover { background: var(--verde-glow); }

.fontes-box {
  margin-top: 6px;
  background: var(--bg2);
  border: 1px solid var(--borda);
  border-radius: 9px;
  padding: 9px 13px;
  display: none;
}

.fontes-box.open { display: block; }

.fonte-item {
  font-size: 11.5px;
  color: var(--texto2);
  padding: 3px 0;
  display: flex;
  align-items: center;
  gap: 7px;
}

.fonte-item::before {
  content: '';
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--verde-dim);
  flex-shrink: 0;
}

/* Typing indicator */
.typing { display: flex; align-items: center; gap: 5px; padding: 12px 16px; }
.typing span {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--verde-dim);
  animation: blink 1.2s ease-in-out infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

/* ─── Barra de input ─── */
.input-bar {
  padding: 14px 28px 20px;
  border-top: 1px solid var(--borda);
  flex-shrink: 0;
  background: var(--bg);
}

.input-inner {
  display: flex;
  align-items: flex-end;
  gap: 9px;
  background: var(--bg2);
  border: 1.5px solid var(--borda2);
  border-radius: 14px;
  padding: 9px 9px 9px 17px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input-inner:focus-within {
  border-color: var(--verde-dim);
  box-shadow: 0 0 0 3px rgba(93,190,133,0.07);
}

.input-inner textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  resize: none;
  font-family: 'Sora', sans-serif;
  font-size: 13.5px;
  color: #D9E8DC;           /* texto claro sobre fundo escuro */
  line-height: 1.55;
  max-height: 130px;
  min-height: 22px;
  overflow-y: auto;
  scrollbar-width: thin;
}

.input-inner textarea::placeholder {
  color: var(--texto3);     /* placeholder mais escuro, ainda legível */
}

.send-btn {
  width: 38px; height: 38px;
  border-radius: 9px;
  background: var(--verde);
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: background 0.14s, transform 0.1s;
}

.send-btn:hover  { background: var(--verde-dim); }
.send-btn:active { transform: scale(0.95); }
.send-btn:disabled { background: var(--surface2); cursor: not-allowed; }

.send-btn svg { width: 16px; height: 16px; fill: #0F1612; }

.input-hint {
  text-align: center;
  font-size: 10.5px;
  color: var(--texto3);
  margin-top: 8px;
}

/* ─── Animações ─── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes blink {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
  40%            { transform: scale(1);   opacity: 1; }
}

/* ─── Responsivo mobile ─── */
@media (max-width: 640px) {
  .sidebar { display: none; }
  .topbar  { padding: 13px 14px 11px; }
  .chat-scroll { padding: 14px 14px 8px; gap: 18px; }
  .input-bar   { padding: 10px 14px 14px; }
  .bubble      { font-size: 13px; }
  .bubble-wrap { max-width: 92%; }
  .welcome h2  { font-size: 22px; }
}
</style>
</head>
<body>
<div class="shell">

  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-row">
        <div class="leaf-icon">
          <svg viewBox="0 0 24 24"><path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 008 20C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/></svg>
        </div>
        <span class="brand-name">RegIA</span>
      </div>
      <div class="brand-tagline">Regulamentações · Universidades Federais</div>
    </div>

    <div class="sidebar-body">
      <div class="sidebar-section">
        <div class="sec-label">Base de dados</div>
        <div class="stats-row">
          <div class="stat-box">
            <div class="stat-num" id="stat-pdfs">—</div>
            <div class="stat-lbl">PDFs</div>
          </div>
          <div class="stat-box">
            <div class="stat-num" id="stat-chunks">—</div>
            <div class="stat-lbl">Chunks</div>
          </div>
        </div>
      </div>

      <hr class="divider">

      <div class="sidebar-section">
        <div class="sec-label">Universidades disponíveis</div>
        <div class="uni-list" id="uni-list"></div>
      </div>

      <hr class="divider">

      <div class="sidebar-section">
        <div class="sec-label">Exemplos de perguntas</div>
        <div class="examples">
          <div class="example-chip" onclick="useExample(this)">A UFPB permite uso de IA em trabalhos acadêmicos?</div>
          <div class="example-chip" onclick="useExample(this)">Compare as políticas da UFMG e da UFRJ sobre IA em provas.</div>
          <div class="example-chip" onclick="useExample(this)">Quais universidades proíbem completamente o uso de IA?</div>
          <div class="example-chip" onclick="useExample(this)">O que a UNIFESP diz sobre plágio com IA?</div>
        </div>
      </div>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <div class="topbar">
      <span class="topbar-title">Consulta de Regulamentações</span>
      <span class="topbar-sub">Respostas baseadas exclusivamente nos documentos oficiais</span>
    </div>

    <div class="chat-scroll" id="chat">
      <div class="welcome" id="welcome">
        <div class="welcome-icon">
          <svg viewBox="0 0 24 24"><path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22l1-2.3A4.49 4.49 0 008 20C19 20 22 3 22 3c-1 2-8 2-12 4 2.5-2.5 7-4.5 9-5.5-2.5.5-5.5 2-7.5 4z"/></svg>
        </div>
        <h2>Olá, sou o RegIA</h2>
        <p>Seu assistente para consultar as regulamentações de uso de Inteligência Artificial nas universidades federais brasileiras. Faça uma pergunta ou escolha um exemplo ao lado.</p>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-inner">
        <textarea id="pergunta" rows="1"
          placeholder="Pergunte sobre regulamentações de IA nas universidades federais..."
          onkeydown="handleKey(event)"
          oninput="autoResize(this)"></textarea>
        <button class="send-btn" id="send-btn" onclick="enviar()">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
      <div class="input-hint">Enter para enviar &nbsp;·&nbsp; Shift+Enter para nova linha</div>
    </div>
  </main>
</div>

<script>
let aguardando = false;

window.addEventListener('message', (e) => {
  if (!e.data) return;
  const d = e.data;

  if (d.type === 'init') {
    document.getElementById('stat-pdfs').textContent = d.n_pdfs;
    const tc = d.total_chunks;
    document.getElementById('stat-chunks').textContent = tc > 999 ? (tc/1000).toFixed(1)+'k' : tc;
    const ul = document.getElementById('uni-list');
    ul.innerHTML = '';
    (d.universidades || []).sort().forEach(u => {
      const el = document.createElement('div');
      el.className = 'uni-tag';
      el.innerHTML = '<span class="uni-dot"></span>' + u;
      el.onclick = () => {
        document.getElementById('pergunta').value = 'Qual a política de IA da ' + u + '?';
        autoResize(document.getElementById('pergunta'));
      };
      ul.appendChild(el);
    });
  }

  if (d.type === 'resposta') {
    removerTyping();
    appendMsg('assistant', d.texto, d.fontes, d.tipo, d.top_k);
    setLoading(false);
  }
});

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); enviar(); }
}

function useExample(el) {
  document.getElementById('pergunta').value = el.textContent.trim();
  autoResize(document.getElementById('pergunta'));
  document.getElementById('pergunta').focus();
}

function scrollBottom() {
  const c = document.getElementById('chat');
  c.scrollTop = c.scrollHeight;
}

function setLoading(v) {
  aguardando = v;
  document.getElementById('send-btn').disabled = v;
}

function esconderWelcome() {
  const w = document.getElementById('welcome');
  if (w) { w.style.display = 'none'; }
}

function appendMsg(role, texto, fontes, tipo, top_k) {
  esconderWelcome();
  const chat = document.getElementById('chat');
  const row  = document.createElement('div');
  row.className = 'msg-row ' + (role === 'user' ? 'user' : 'bot');

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + (role === 'user' ? 'user' : 'bot');
  avatar.textContent = role === 'user' ? 'VC' : 'RI';

  const wrap   = document.createElement('div');
  wrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + (role === 'user' ? 'user' : 'bot');
  bubble.textContent = texto;
  wrap.appendChild(bubble);

  if (role === 'assistant') {
    const meta = document.createElement('div');
    meta.className = 'meta';

    if (tipo) {
      const tag = document.createElement('span');
      tag.className = 'meta-tag';
      const labels = { especifica: 'consulta específica', comparativa: 'consulta comparativa', geral: 'consulta geral' };
      tag.textContent = (labels[tipo] || tipo) + (top_k ? ' · ' + top_k + ' chunks' : '');
      meta.appendChild(tag);
    }

    if (fontes && fontes.length) {
      const btn = document.createElement('button');
      btn.className = 'fontes-toggle';
      btn.textContent = fontes.length + ' fonte' + (fontes.length > 1 ? 's' : '');

      const box = document.createElement('div');
      box.className = 'fontes-box';
      fontes.forEach(f => {
        const item = document.createElement('div');
        item.className = 'fonte-item';
        item.textContent = f;
        box.appendChild(item);
      });

      btn.onclick = () => box.classList.toggle('open');
      meta.appendChild(btn);
      wrap.appendChild(meta);
      wrap.appendChild(box);
    } else {
      wrap.appendChild(meta);
    }
  }

  row.appendChild(avatar);
  row.appendChild(wrap);
  chat.appendChild(row);
  scrollBottom();
}

function adicionarTyping() {
  esconderWelcome();
  const chat = document.getElementById('chat');
  const row  = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'typing-row';

  const avatar = document.createElement('div');
  avatar.className = 'avatar bot';
  avatar.textContent = 'RI';

  const wrap   = document.createElement('div');
  wrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = 'bubble bot';
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  wrap.appendChild(bubble);

  row.appendChild(avatar);
  row.appendChild(wrap);
  chat.appendChild(row);
  scrollBottom();
}

function removerTyping() {
  const t = document.getElementById('typing-row');
  if (t) t.remove();
}

function enviar() {
  if (aguardando) return;
  const inp   = document.getElementById('pergunta');
  const texto = inp.value.trim();
  if (!texto) return;

  appendMsg('user', texto);
  inp.value = '';
  inp.style.height = 'auto';
  adicionarTyping();
  setLoading(true);

  window.parent.postMessage({ type: 'pergunta', texto }, '*');
}
</script>
</body>
</html>"""

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
    """
    Le os PDFs da pasta /pdfs, indexa e carrega todos os modelos.
    Executado uma unica vez por instancia; resultado fica em cache.
    """
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # Modelos
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

    # Leitura dos PDFs
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

    # Chunking
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

    # Indexacao
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
# Streamlit — orquestrador mínimo
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RegIA · Regulamentações de IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove todo o chrome nativo do Streamlit
st.markdown("""
<style>
  #MainMenu, header, footer,
  [data-testid="stToolbar"],
  [data-testid="stSidebar"],
  [data-testid="collapsedControl"] { display: none !important; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  .stApp { background: #0F1612 !important; }
  iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# Carrega pipeline uma única vez
with st.spinner("🌱 Carregando modelos e indexando documentos..."):
    chroma_store, bm25, reranker_model, llm, contagem, n_pdfs = construir_pipeline()

# Renderiza a UI customizada num iframe que ocupa a tela toda
components.html(HTML_UI, height=720, scrolling=False)

# Injeta os dados da sidebar no iframe após renderizar
dados_init = {
    "type":          "init",
    "n_pdfs":        n_pdfs,
    "total_chunks":  sum(contagem.values()),
    "universidades": sorted(contagem.keys()),
}

st.markdown(f"""
<script>
(function injetar() {{
  const frame = Array.from(document.querySelectorAll('iframe'))
                     .find(f => f.contentDocument &&
                                f.contentDocument.getElementById('chat'));
  if (frame) {{
    frame.contentWindow.postMessage({json.dumps(dados_init)}, '*');
  }} else {{
    setTimeout(injetar, 300);
  }}
}})();
</script>
""", unsafe_allow_html=True)

# Bridge: captura mensagem do iframe via query param e processa
pergunta_qp = st.query_params.get("q", None)
ultima_q    = st.session_state.get("ultima_q", None)

if pergunta_qp and pergunta_qp != ultima_q:
    st.session_state["ultima_q"] = pergunta_qp

    with st.spinner("Consultando documentos..."):
        resultado = responder(chroma_store, bm25, reranker_model, llm, pergunta_qp)

    dados_resp = {
        "type":   "resposta",
        "texto":  resultado["resposta"],
        "fontes": resultado["fontes"],
        "tipo":   resultado["tipo"],
        "top_k":  resultado["top_k"],
    }

    st.markdown(f"""
<script>
(function enviar() {{
  const frame = Array.from(document.querySelectorAll('iframe'))
                     .find(f => f.contentDocument &&
                                f.contentDocument.getElementById('chat'));
  if (frame) {{
    frame.contentWindow.postMessage({json.dumps(dados_resp)}, '*');
  }} else {{
    setTimeout(enviar, 300);
  }}
}})();
</script>
""", unsafe_allow_html=True)

# Bridge JS: intercepta postMessage do iframe e dispara rerun via query param
st.markdown("""
<script>
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'pergunta') {
    const url = new URL(window.location.href);
    url.searchParams.set('q', e.data.texto);
    window.history.replaceState({}, '', url.toString());
    window.location.reload();
  }
});
</script>
""", unsafe_allow_html=True)
