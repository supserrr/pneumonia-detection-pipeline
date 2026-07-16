FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/

# Run as a non-root user (UID 1000). Hugging Face Spaces runs containers as this
# user, and owning /app lets the /upload and /retrain endpoints write into
# data/ and models/ at runtime. Also good practice on Render and locally.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
