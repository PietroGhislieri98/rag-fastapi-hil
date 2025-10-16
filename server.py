import os, uuid
from typing import List, Dict, Any, Optional, TypedDict

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver

# --------- Env ---------
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
CHAT_MODEL   = os.getenv("CHAT_MODEL", "mistral:7b-instruct")
CHROMA_HOST  = os.getenv("CHROMA_HOST", "localhost").replace("http://","").replace("https://","").strip("/")
CHROMA_PORT  = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION   = os.getenv("CHROMA_COLLECTION", "docs")
PG_DSN       = os.getenv("LG_PG_DSN", "postgresql://rag:ragpass@localhost:5432/rag_db?sslmode=disable")

RAG_PROMPT = PromptTemplate.from_template(
"""Sei un assistente RAG.
Usa SOLO il contesto per rispondere e cita nel formato [source#chunk].
Se l'informazione non è presente, dillo.

Domanda: {question}

Contesto:
{context}

Risposta:"""
)

app = FastAPI(title="RAG API (FastAPI + LangGraph + Chroma + Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------- Models ---------
class IngestRequest(BaseModel):
    doc_id: str
    text: str
    chunk_size: int = 800
    chunk_overlap: int = 120

class AskStartRequest(BaseModel):
    question: str
    topk: int = 4
    thread_id: Optional[str] = None

class AskResumeRequest(BaseModel):
    thread_id: str
    decision: Dict[str, Any]

# --------- Helpers ---------
def embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

def vectordb(emb):
    http = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return Chroma(client=http, collection_name=COLLECTION, embedding_function=emb)

# --------- Health ---------
@app.get("/health")
def health():
    try:
        c = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        hb = c.heartbeat()
    except Exception as e:
        raise HTTPException(503, f"Chroma non raggiungibile: {e}")
    try:
        _ = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL)
    except Exception as e:
        raise HTTPException(503, f"Ollama non raggiungibile: {e}")
    return {"chroma": hb, "ollama": "ok"}

# --------- /ingest ---------
@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(400, "Campo 'text' vuoto")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap
    )
    chunks = splitter.split_text(req.text)
    metas  = [{"source": req.doc_id, "chunk": i} for i in range(len(chunks))]
    ids    = [f"{req.doc_id}-{i}-{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    emb = embeddings()
    db  = vectordb(emb)
    db.add_texts(texts=chunks, metadatas=metas, ids=ids)
    return {"inserted": len(chunks), "collection": COLLECTION, "doc_id": req.doc_id}

# --------- LangGraph (retrieve -> context -> human -> generate) ---------
class AState(TypedDict, total=False):
    question: str
    topk: int
    retrieved: List[Dict[str, Any]]
    context: str
    answer: str

def retrieve_node(s: AState) -> AState:
    emb = embeddings(); db = vectordb(emb)
    k = int(s.get("topk", 4))
    docs = db.similarity_search(s["question"], k=k)
    got = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    return {"retrieved": got}

def context_node(s: AState) -> AState:
    parts=[]
    for d in s.get("retrieved", []):
        src = d["metadata"].get("source","?"); ch = d["metadata"].get("chunk","?")
        parts.append(f"[{src}#{ch}]\n{d['page_content'].strip()}")
    return {"context": "\n\n".join(parts)}

def human_node(s: AState) -> AState:
    payload = {
        "action": "review_context",
        "question": s.get("question"),
        "context_preview": s.get("context","")[:4000],
        "retrieved_sources": [
            {"source": d["metadata"].get("source"), "chunk": d["metadata"].get("chunk")}
            for d in s.get("retrieved", [])
        ],
        "hint": "Rispondi con {'approved': true} oppure {'approved': false, 'edited_context': '...'}"
    }
    decision = interrupt(payload)  # pausa qui
    if isinstance(decision, dict) and decision.get("edited_context"):
        return {"context": str(decision["edited_context"])}
    return {}  # lascia il contesto com'è

def generate_node(s: AState) -> AState:
    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL)
    prompt = RAG_PROMPT.format(question=s["question"], context=s.get("context",""))
    msg = llm.invoke(prompt)
    return {"answer": msg.content}

# Checkpointer Postgres
PG = PostgresSaver.from_conn_string(PG_DSN)
PG.setup()

sg = StateGraph(AState)
sg.add_node("retrieve", retrieve_node)
sg.add_node("context", context_node)
sg.add_node("human",   human_node)
sg.add_node("generate", generate_node)
sg.add_edge(START, "retrieve"); sg.add_edge("retrieve","context")
sg.add_edge("context","human"); sg.add_edge("human","generate"); sg.add_edge("generate", END)
GRAPH = sg.compile(checkpointer=PG)

# --------- /ask/start ---------
@app.post("/ask/start")
def ask_start(req: AskStartRequest):
    tid = req.thread_id or uuid.uuid4().hex
    out = GRAPH.invoke({"question": req.question, "topk": req.topk},
                       config={"configurable": {"thread_id": tid}})
    if isinstance(out, dict) and "__interrupt__" in out:
        return {"thread_id": tid, "interrupt": out["__interrupt__"]}
    return {"thread_id": tid, "answer": out["answer"]}

# --------- /ask/resume ---------
@app.post("/ask/resume")
def ask_resume(req: AskResumeRequest):
    out = GRAPH.invoke(Command(resume=req.decision),
                       config={"configurable": {"thread_id": req.thread_id}})
    if isinstance(out, dict) and "__interrupt__" in out:
        return {"thread_id": req.thread_id, "interrupt": out["__interrupt__"]}
    return {"thread_id": req.thread_id, "answer": out["answer"]}
