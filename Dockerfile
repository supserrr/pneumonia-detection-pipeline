# ---- Stage 1: build the React / shadcn (neobrutalism) frontend ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/ ./frontend/
# vite build outputs to ../app/static (see frontend/vite.config.ts) => /fe/app/static
RUN cd frontend && npm ci --no-audit --no-fund && npm run build

# ---- Stage 2: Python API that serves the built SPA ----
FROM python:3.10-slim
WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/
# freshly built SPA (index.html + assets) into app/static
COPY --from=frontend /fe/app/static/ app/static/

# Run as a non-root user (UID 1000): lets /upload and /retrain write into
# data/ and models/ at runtime. Good practice on Render/Spaces and locally.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
