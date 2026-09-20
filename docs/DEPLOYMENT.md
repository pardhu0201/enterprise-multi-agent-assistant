# Deployment

The whole app ships as **one Docker image** — the React bundle is built in a
first stage and served by FastAPI as static files in the second, so the
public demo is one container with one URL.

## Recommended: Render (free web service, SQLite)

This is the $0 default. Hugging Face Docker Spaces now require a paid PRO
subscription even on the free `cpu-basic` tier (confirmed live — a plain
Docker Space creation returns `402 Payment Required` without PRO), so they are
no longer a free option; see the HF Spaces section below if you do have PRO.

1. Click **[Deploy to Render](https://render.com/deploy?repo=https://github.com/pardhu0201/enterprise-multi-agent-assistant)**
   (also linked from the repo README), or in the Render dashboard choose
   **New → Blueprint** and point it at this GitHub repo. Render reads
   `render.yaml` at the repo root automatically.
2. Render provisions one free Docker web service from the root `Dockerfile`.
   No database is provisioned — the app runs on SQLite baked into the
   container's disk, which is this app's zero-setup default and has no
   expiry (Render's free *Postgres*, by contrast, now expires 30 days after
   creation, which is why the blueprint deliberately doesn't provision one).
3. Optionally add an `ANTHROPIC_API_KEY` secret in the service's **Environment**
   tab to switch on Claude reasoning. Nothing else is required.
4. Render's free tier **sleeps after 15 minutes of inactivity** and takes
   roughly 50 seconds to wake on the next request — expected for a portfolio
   demo, not a bug. Health check is `/api/health`.

To use a longer-lived Postgres store instead of SQLite (e.g. so data survives
a Render service rebuild, or to exercise the native-pgvector code path), add a
free [Neon](https://neon.tech) project and set `DATABASE_URL` to its pooled
connection string in the service's environment — no code or image change
needed.

## Alternative: Hugging Face Spaces (requires HF PRO)

Hugging Face's Docker Spaces do not sleep, which makes them a better host
*if* you already have a PRO subscription (currently $9/mo).

1. Create a new Space → **Docker** SDK, public visibility.
2. Copy `deploy/huggingface/README.md`'s YAML front-matter to the Space's own
   `README.md`, or connect the Space to this GitHub repo via HF's GitHub-sync
   option.
3. Add `ANTHROPIC_API_KEY` as a Space secret if you want Claude reasoning.
4. The Space builds the root `Dockerfile` and exposes port `7860` (already
   set as `app_port` in the front-matter and as the container's default).

## Alternative: Vercel (frontend) + Render (backend only)

Use this shape if you want the frontend and backend on separate hosts (e.g.
a custom domain on Vercel) rather than the single-container deploy above.

1. Deploy the same root `Dockerfile` to Render (via the blueprint above),
   but set `SERVE_FRONTEND=false` in the service's environment — the built
   frontend is still baked into the image, but FastAPI stops serving it and
   only answers under `/api/*`, which is all a separately-hosted UI needs.
2. Deploy `frontend/` to Vercel using the build settings in
   `frontend/vercel.json` (already committed), with:
   ```
   VITE_API_BASE=https://<your-render-service>.onrender.com
   ```
3. Vercel's preview/prod domains are already allow-listed in the backend's
   CORS config (`allow_origin_regex` matches `*.vercel.app`); add a custom
   domain to `CORS_ORIGINS` if you use one.

## Local: Docker Compose (Postgres + pgvector, mirrors a production shape)

```bash
docker compose up --build
# open http://localhost:7860
```

`docker-compose.yml` runs `pgvector/pgvector:pg17` alongside the app so the
native-vector code path is exercised locally, not just the numpy fallback.

## Environment variables

Every variable has a working default — see `.env.example` at the repo root
for the full list with inline documentation. The ones that matter for a
deployment:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Blank = deterministic demo mode (free). Set to enable Claude reasoning. |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Only change if instructed to use a different model. |
| `DATABASE_URL` | *(empty → SQLite)* | Postgres connection string (Neon or any). `postgres://` and `postgresql://` are both normalised to the psycopg3 driver automatically. |
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
