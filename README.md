# Qdrant Recommendation (Python, self-hosted)

Minimal project to store **campaign embeddings** in **Qdrant** and get **personalized recommendations** from a user's last-10 donations.

## What you get
- Local **Qdrant** via Docker
- Clean HTML → canonical text
- Embed with **Sentence Transformers** (multilingual, 384-dim)
- Upsert/search in Qdrant with filters
- Simple **user profile vector** (avg of last-10 donated campaigns)
- Demo CLI that:
  1) Ingests sample campaigns → Qdrant
  2) Computes recommendations for a sample user

---

## Quick start

### 1) Run Qdrant
```bash
docker compose up -d
```

### 2) Create venv & install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 3) Ingest sample data
```bash
python -m src.app.demo ingest
```

### 4) Get recommendations
```bash
python -m src.app.demo recommend --user-id 1001 --top-k 5
```

You should see top campaigns (by ID + score).

---

## Project structure
```
.
├─ docker-compose.yml
├─ requirements.txt
├─ .env.example  (.env after you copy)
├─ data/
│  ├─ campaigns.json     # sample campaigns with HTML-ish descriptions
│  └─ donations.json     # last-10 donations per user
└─ src/app/
   ├─ clean.py
   ├─ embeddings.py
   ├─ qdrant_store.py
   ├─ recommend.py
   └─ demo.py
```

---

## Swap the model
Default is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim).
Change `EMBEDDING_MODEL` in `.env` if you want another model. Ensure **Qdrant collection vector size** matches the model.

---

## Notes
- This demo uses local JSON instead of Postgres. Replace the loaders with your queries.
- For production, add:
  - Metadata filters (e.g., `is_active`, `country`, `category_id`)
  - Business re-rank (freshness, urgency, traction)
  - A cache (e.g., Redis)
  - Batch upserts & monitoring

---

## Go migration

A Go module now lives under `go/` with an HTTP client that mirrors the Python helper functions for creating collections, upserting points, and running similarity searches. See `docs/go_migration_plan.md` for the broader migration strategy and `go/cmd/demo` for a minimal usage example.
