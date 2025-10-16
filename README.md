# RAG FastAPI HIL (LangGraph + Chroma + Ollama + Postgres)

API RAG con **human-in-the-loop** (interrupt/resume) basata su **FastAPI**, **LangGraph** (checkpointer **Postgres**), **Chroma** (vector DB) e **Ollama** (LLM + embeddings).

* Endpoint chiave: `/ingest`, `/ask/start`, `/ask/resume`
* HITL: revisione/approvazione del contesto prima della generazione
* Storage persistente delle interruzioni via **PostgresSaver**

---

## 1) Requisiti & Installazione (Linux)

### 1.1 Requisiti minimi

* Ubuntu/Debian recenti
* **Docker Engine** + **Docker Compose v2**
* **Python 3.11+** (solo se vuoi avviare l’API in locale, non in container)
* ~8–12 GB RAM consigliati (per modelli LLM locali)
* Porte libere: `5432` (Postgres), `8000` (Chroma), `11434` (Ollama), `9000` (API)

---

### 1.2 Installazione Docker & Compose (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
# (opzione rapida)
sudo apt-get install -y docker.io docker-compose-plugin
# abilita e usa docker da non-root
sudo usermod -aG docker $USER
# logout/login per applicare il gruppo
```

> In alternativa configura il repo ufficiale Docker. L’importante è avere `docker` e `docker compose` disponibili.

---

### 1.3 Installazione Python 3.11 (opzionale, per avvio locale)

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

Creazione venv (quando servirà):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

### 1.4 Installazione componenti AI

#### A) **Ollama** (LLM + embeddings)

**Opzione consigliata:** usare il container nel `docker-compose` (vedi più sotto).
**Opzione host (se vuoi Ollama fuori da Docker):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
# scarica i modelli necessari (embedding + chat)
ollama pull nomic-embed-text:latest
ollama pull mistral:7b-instruct
```

> Se usi l’Ollama host, l’API ascolta su `http://localhost:11434`.

#### B) **Chroma** (vector DB) — via Docker

```bash
docker run -d --name chroma --restart unless-stopped \
  -p 8000:8000 \
  -v /srv/chroma:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:0.6.3

# health check
curl http://localhost:8000/api/v2/heartbeat
```

#### C) **Postgres** — via Docker

```bash
docker run -d --name pg-rag --restart unless-stopped \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=ragpass \
  -e POSTGRES_USER=rag \
  -e POSTGRES_DB=rag_db \
  -v /srv/pgdata:/var/lib/postgresql/data \
  postgres:16
```

---

### 1.5 Variabili d’ambiente

Copia `.env.example` in `.env` e regola i valori:

```bash
cp .env.example .env
```

`./.env` (modalità **Docker Compose**, servizi in rete interna):

```
# LLM & Embeddings (Ollama in container)
OLLAMA_URL=http://ollama:11434
EMBED_MODEL=nomic-embed-text:latest
CHAT_MODEL=mistral:7b-instruct

# Chroma (server HTTP)
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_COLLECTION=docs

# LangGraph checkpointer (Postgres)
LG_PG_DSN=postgresql://rag:ragpass@pg-rag:5432/rag_db?sslmode=disable

# API
API_PORT=9000
```

> Se vuoi avviare l’API **in locale** (fuori da Docker), imposta:
>
> * `OLLAMA_URL=http://localhost:11434`
> * `CHROMA_HOST=localhost`
> * `LG_PG_DSN=postgresql://rag:ragpass@localhost:5432/rag_db?sslmode=disable`

---

### 1.6 Dipendenze Python (solo per avvio locale)

```bash
# nella root del progetto, con venv attiva
pip install -r requirements.txt
```

---

## 2) Avvio del progetto (dopo il clone)

```bash
git clone https://github.com/<tuo-org>/rag-fastapi-hil.git
cd rag-fastapi-hil
cp .env.example .env
```

### 2.1 Avvio con **Docker Compose** (consigliato)

```bash
docker compose up -d
```

Scarica i modelli nell’istanza **Ollama** del compose:

```bash
# identifica il container (nome può variare)
docker compose ps
# poi:
docker compose exec ollama ollama pull nomic-embed-text:latest
docker compose exec ollama ollama pull mistral:7b-instruct
```

Verifica salute servizi:

```bash
curl http://localhost:9000/health
# atteso: {"chroma": "...", "ollama": "ok"}
```

### 2.2 Avvio **locale** dell’API (alternativa per sviluppo)

Assicurati che **Chroma** e **Postgres** siano in esecuzione (anche via Docker), e che **Ollama** sia disponibile (host o container).

```bash
# avvia le dipendenze con i container (se non già avviati)
docker run -d --name pg-rag --restart unless-stopped -p 5432:5432 \
  -e POSTGRES_PASSWORD=ragpass -e POSTGRES_USER=rag -e POSTGRES_DB=rag_db \
  -v /srv/pgdata:/var/lib/postgresql/data postgres:16

docker run -d --name chroma --restart unless-stopped -p 8000:8000 \
  -v /srv/chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:0.6.3

# (se Ollama host)
sudo systemctl start ollama
ollama pull nomic-embed-text:latest
ollama pull mistral:7b-instruct

# attiva venv e avvia FastAPI
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

uvicorn server:app --host 0.0.0.0 --port 9000 --reload
```

---

## 3) API — Uso rapido

### 3.1 Health

```bash
curl http://localhost:9000/health
```

### 3.2 Ingest (chunking + embedding + store in Chroma)

```bash
curl -X POST http://localhost:9000/ingest \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "doc_id": "guide1",
  "text": "Questo è un testo di prova su HPA e VPA ...",
  "chunk_size": 800,
  "chunk_overlap": 120
}
JSON
```

**Risposta:**

```json
{"inserted": N, "collection": "docs", "doc_id": "guide1"}
```

### 3.3 Ask (fase 1: start ⇒ possibile interrupt)

```bash
curl -s -X POST http://localhost:9000/ask/start \
  -H 'Content-Type: application/json' \
  -d '{"question":"Differenza tra HPA e VPA","topk":4}' | jq
```

**Possibili risposte:**

* Interrupt (HITL):

  ```json
  {
    "thread_id": "abc123...",
    "interrupt": { "value": { "action": "review_context", "context_preview": "...", "retrieved_sources": [...] } }
  }
  ```
* Risposta diretta:

  ```json
  { "thread_id": "abc123...", "answer": "..." }
  ```

### 3.4 Ask (fase 2: resume)

**Approva contesto:**

```bash
curl -s -X POST http://localhost:9000/ask/resume \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"abc123...", "decision":{"approved":true}}' | jq
```

**Modifica contesto:**

```bash
curl -s -X POST http://localhost:9000/ask/resume \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "thread_id": "abc123...",
  "decision": {
    "approved": false,
    "edited_context": "[guide1#0]\n...solo i pezzi che vuoi..."
  }
}
JSON
```

---

## 4) Varianti d’uso & Note

* **Stesso modello di embedding** in ingest e query (default: `nomic-embed-text:latest`).
* **CORS** è abilitato per facilitare l’uso da frontend/laravel.
* **Persistenza interrupt**: `PostgresSaver` crea le tabelle alla prima esecuzione (`PG.setup()`).
* **Streaming**: se vuoi risposta token-by-token, si può adattare `generate_node` usando `.stream()` e una `StreamingResponse`.

---

## 5) Troubleshooting

* **`/health` fallisce su Chroma** → verifica container, versione `0.6.x` e route `/api/v2/heartbeat`.
* **Mismatch embedding** → se cambi modello (es. da `nomic-embed-text` a altro), re-ingesta i documenti.
* **Ollama lento/all’avvio** → il primo prompt compila/ottimizza; poi si stabilizza.
* **`host.docker.internal` su Linux** → usa **nomi servizio** del compose (es. `ollama`, `chroma`, `pg-rag`) oppure l’IP della network Docker.
* **Permessi Docker** → se non sei nel gruppo `docker`, aggiungiti e rifai login.

---

## 6) Struttura del repo

```
rag-fastapi-hil/
├─ server.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ .gitignore
└─ README.md  ← questo file
```

---

## 7) Licenza


---

