# ─── Stage 1 : installation des dépendances Python ───────────────────────────
# On utilise uv pour respecter exactement uv.lock (reproductible).
# torch est installé depuis le dépôt CUDA d'origine (Linux x86_64) ; il
# fonctionne aussi bien sur CPU — aucun GPU n'est requis.
FROM python:3.13-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copier uniquement les manifestes pour maximiser le cache de couche
COPY pyproject.toml uv.lock ./

# Installer toutes les dépendances dans /app/.venv (--frozen = uv.lock exact)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ─── Stage 2 : image de production minimale ───────────────────────────────────
FROM python:3.13-slim

# Bibliothèques système requises par OpenCV (headless) et PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le venv construit dans le stage précédent
COPY --from=builder /app/.venv /app/.venv

# Copier le code source
COPY src/ ./src/

# Répertoires montés en volume à l'exécution ; les créer vides ici
# garantit que le démarrage ne plante pas si les volumes ne sont pas montés.
RUN mkdir -p outputs/api \
             runs/detect/models/yolo_impacts/weights

# Activer le venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Vérification de santé : appel à /health toutes les 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
