import os
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, status, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

app = FastAPI(title="ShootScore Upload API", version="1.0.0")
logger = logging.getLogger(__name__)

# Configuration
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN", "changeme-very-secure-token")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/data/uploads"))
DB_PATH = Path(os.getenv("DB_PATH", "/app/db/shootscore.db"))

# Créer les répertoires
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Logger
logging.basicConfig(level=logging.INFO)


# ── Base de données ────────────────────────────────────────────────────────

def init_db():
    """Initialise la base de données SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            shot_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            checksum TEXT UNIQUE NOT NULL,
            filesize INTEGER,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def get_file_checksum(file_data: bytes) -> str:
    """Calcule le checksum SHA256 d'un fichier."""
    return hashlib.sha256(file_data).hexdigest()


def check_duplicate(checksum: str) -> bool:
    """Vérifie si un checksum existe déjà en base."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM uploads WHERE checksum = ?", (checksum,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def record_upload(series_id: str, shot_id: str, filename: str, checksum: str, filesize: int, metadata: str = None):
    """Enregistre un upload en base de données."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO uploads (series_id, shot_id, filename, checksum, filesize, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (series_id, shot_id, filename, checksum, filesize, metadata))
        conn.commit()
        logger.info(f"Recorded upload: {series_id}/{shot_id} checksum={checksum}")
    except sqlite3.IntegrityError:
        logger.warning(f"Duplicate checksum detected: {checksum}")
        raise
    finally:
        conn.close()


# ── Authentification ───────────────────────────────────────────────────────

def verify_token(authorization: Optional[str] = Header(None)) -> bool:
    """Vérifie le token Bearer."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization header")
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header")
    
    token = parts[1]
    if token != API_SECRET_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    return True


# ── Routes ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialise la DB au démarrage."""
    init_db()


@app.get("/health")
async def health():
    """Endpoint de health check."""
    return {"status": "ok"}


class UploadMetadata(BaseModel):
    series_id: str
    shot_id: str


@app.post("/api/v1/upload")
async def upload_image(
    file: UploadFile = File(...),
    series_id: str = Form(...),
    shot_id: str = Form(...),
    authorization: Optional[str] = Header(None),
):
    """
    Endpoint d'upload d'images brutes.
    
    - Reçoit une image JPEG
    - Vérifie le token Bearer
    - Calcule le checksum pour déduplication
    - Stocke l'image par date
    - Enregistre en base de données
    """
    
    # Authentification
    verify_token(authorization)
    
    try:
        # Lire le fichier
        file_data = await file.read()
        if not file_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
        
        filesize = len(file_data)
        
        # Calculer le checksum
        checksum = get_file_checksum(file_data)
        
        # Vérifier les doublons
        if check_duplicate(checksum):
            logger.warning(f"Duplicate image rejected: {series_id}/{shot_id} (checksum={checksum})")
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"detail": "Duplicate image (already uploaded)", "checksum": checksum}
            )
        
        # Créer le répertoire de la date
        today = datetime.now().strftime("%Y-%m-%d")
        day_dir = UPLOAD_DIR / today
        day_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le fichier
        filename = f"{series_id}_{shot_id}.jpg"
        filepath = day_dir / filename
        
        with open(filepath, "wb") as f:
            f.write(file_data)
        
        # Enregistrer en base
        record_upload(series_id, shot_id, str(filepath), checksum, filesize, file.content_type)
        
        logger.info(f"Upload successful: {series_id}/{shot_id} ({filesize} bytes) -> {filepath}")
        
        return {
            "status": "success",
            "filename": filename,
            "checksum": checksum,
            "filesize": filesize,
            "stored_at": str(filepath),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/")
async def root():
    """Endpoint racine."""
    return {"message": "ShootScore Upload API v1.0"}
