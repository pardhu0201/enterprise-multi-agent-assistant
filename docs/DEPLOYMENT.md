# Deployment

The whole app ships as **one Docker image** — the React bundle is built in a
first stage and served by FastAPI as static files in the second, so the
public demo is one container with one URL. Everything below is a free tier.

## Recommended: Hugging Face Spaces (backend + UI) + no separate frontend host

Hugging Face's free Docker Spaces do not sleep on inactivity, which is why
this is the recommended host for a demo people will actually click into.

1. Create a new Space → **Docker** SDK, public visibility.
2. Push this repo's contents to the Space's git remote (`deploy/huggingface/README.md`
   has the Space's YAML front-matter already — copy it to the repo root as the
   Space's `README.md`, or point the Space at this repo directly if using the
   GitHub-sync option).
3. In the Space's **Settings → Repository secrets**, optionally add
   `ANTHROPIC_API_KEY` to switch on Claude reasoning. No other configuration
   is required — SQLite persists inside the container's writable layer, and
   the seed corpus re-ingests idempotently on every boot.
4. The Space builds `Dockerfile` at the repo root and exposes port `7860`
   (already set as `app_port` in the front-matter and as the container's
   `EXPOSE`/`PORT` default).

For a persistent database across Space restarts (rather than SQLite baked
into the container layer), add a Postgres connection string as the
`DATABASE_URL` secret — see the Neon option below.

## Alternative: Render (backend) + Vercel (frontend) + Neon (Postgres)

Use this shape when you want the "full-stack, three services" story on a
resume, or when HF Spaces isn't an option.

### 1. Database — Neon (free Postgres + pgvector)

1. Create a project at [neon.tech](https://neon.tech).
2. Neon's default images include the `vector` extension — no action needed;
   the app runs `CREATE EXTENSION IF NOT EXISTS vector` on boot.
3. Copy the pooled connection string (the `-pooler` host). That's `DATABASE_URL`.

### 2. Backend — Render (free Docker web service)

1. Push this repo to GitHub, then **New → Blueprint** in Render and point it
   at `deploy/render.yaml`. It provisions the web service and (if you don't
   use Neon) a free Render Postgres instead — edit the blueprint's `databases`
   block to remove that if you're using Neon.
2. Set the `ANTHROPIC_API_KEY` secret (optional) and, if using Neon, replace
   the blueprint's `DATABASE_URL` binding with your Neon connection string.
3. Render's free tier **sleeps after 15 minutes of inactivity** and takes
   roughly 50 seconds to wake — acceptable for a portfolio link, not for a
   latency-sensitive demo. Health check is `/api/health`.

### 3. Frontend — Vercel

Only needed if you deploy the backend **without** baking in the frontend
(i.e. you deployed `backend/` alone to Render and want a separately-hosted
UI). Point Vercel at `frontend/`, set the build command from
`frontend/vercel.json` (already committed), and set the environment variable:

```
VITE_API_BASE=https://<your-render-service>.onrender.com
```

Vercel's preview/prod domains are already allow-listed in the backend's CORS
config (`allow_origin_regex` matches `*.vercel.app`); for a custom domain add
it to `CORS_ORIGINS` in the backend's environment.

## Local: Docker Compose (Postgres + pgvector, mirrors production)

```bash
docker compose up --build
# open http://localhost:7860
```

`docker-compose.yml` runs `pgvector/pgvector:pg17` alongside the app so the
native-vector code path is exercised locally, not just numpy fallback.

## Environment variables

Every variable has a working default — see `.env.example` at the repo root
for the full list with inline documentation. The ones that matter for a
deployment:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Blank = deterministic demo mode (free). Set to enable Claude reasoning. |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Only change if instructed to use a different model. |
| `DATABASE_URL` | *(empty → SQLite)* | Postgres connection string (Neon/Render/local). `postgres://` and `postgresql://` are both normalised to the psycopg3 driver automatically. |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (zero-dependency) or `sentence-transformers` (better recall, heavier image). |
| `CORS_ORIGINS` | `localhost:5173` | Comma-separated list; add your frontend's production origin. |
| `SEED_ON_STARTUP` | `true` | Re-ingests the bundled policy corpus on every boot (idempotent — skips unchanged documents). |

## Verifying a deployment

```bash
curl https://<your-host>/api/health
# {"status":"ok","llm_mode":"demo"|"claude","documents":9,"chunks":88,...}
```

`llm_mode` confirms whether the API key was picked up; `documents`/`chunks`
confirm the seed corpus ingested successfully.
