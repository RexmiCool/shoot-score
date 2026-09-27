# ShootScore Upload Server

API FastAPI pour recevoir et stocker les images brutes des utilisateurs ShootScore en vue de l'entraînement continu.

## Fonctionnalités

- 📸 **POST endpoint** : Reçoit les images brutes en multipart/form-data
- 🔒 **Authentification** : Token Bearer
- 💾 **Stockage local** : Organisé par date (`uploads/YYYY-MM-DD/`)
- 🔍 **Déduplication** : Checksum SHA256 pour éviter les doublons
- 📊 **Indexation** : Base de données SQLite avec métadonnées
- 🐳 **Docker** : Containerisée, prête pour RPi
- 🚀 **Asynchrone** : Non-bloquant côté upload

## Structure

```
server/
├── Dockerfile              # Image Docker
├── docker-compose.yml      # Orchestration
├── requirements.txt        # Dépendances Python
├── .env.example           # Configuration (à copier en .env)
├── .gitignore             # Exclusions Git
├── app/
│   ├── __init__.py
│   └── main.py            # Application FastAPI
├── data/
│   └── uploads/           # Stockage des images
│       ├── 2026-09-27/
│       └── 2026-09-28/
└── db/
    └── shootscore.db      # Base de données SQLite
```

## Installation rapide

### Sur Raspberry Pi

```bash
# 1. Cloner le repo
git clone <repo> /home/pi/shootscore
cd /home/pi/shootscore/server

# 2. Créer le .env
cp .env.example .env
nano .env  # Remplacer API_SECRET_TOKEN

# 3. Lancer avec Docker
docker-compose up -d

# 4. Vérifier
docker-compose logs -f
curl http://localhost:8000/health
```

## API

### POST /api/v1/upload

Endpoint principal pour recevoir les images.

**Headers** :
```
Authorization: Bearer <TOKEN>
```

**Paramètres** :
- `file` : Fichier JPEG (multipart/form-data)
- `series_id` : Identifiant de la série (string)
- `shot_id` : Identifiant du tir (string)

**Réponses** :
- `200 OK` : Image uploadée avec succès
- `201 Created` : Image créée
- `409 Conflict` : Doublon détecté (checksum exists)
- `401 Unauthorized` : Token manquant ou invalide
- `400 Bad Request` : Paramètres manquants
- `500 Internal Server Error` : Erreur serveur

**Exemple** :
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "Authorization: Bearer your-token" \
  -F "file=@photo.jpg" \
  -F "series_id=123" \
  -F "shot_id=0"
```

### GET /health

Santé du serveur.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Configuration

### .env

```env
# Token d'authentification Bearer (min. 32 caractères)
API_SECRET_TOKEN=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0

# Répertoire de stockage (utilisé par Docker)
UPLOAD_DIR=/app/data/uploads

# Chemin base de données SQLite
DB_PATH=/app/db/shootscore.db
```

## Base de données

### Schéma

```sql
CREATE TABLE uploads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  series_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  checksum TEXT UNIQUE NOT NULL,
  filesize INTEGER,
  uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  metadata TEXT
);
```

### Requêtes utiles

```bash
sqlite3 db/shootscore.db

# Nombre total d'uploads
SELECT COUNT(*) FROM uploads;

# Uploads par série
SELECT series_id, COUNT(*) FROM uploads GROUP BY series_id;

# Derniers uploads
SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT 10;

# Espace utilisé par série
SELECT series_id, SUM(filesize) as total_bytes FROM uploads GROUP BY series_id;
```

## Sécurité

| Aspect | Détails |
|--------|---------|
| **Authentification** | Bearer token, vérifié sur chaque requête |
| **Autorisations** | POST seul, pas de GET (prévient téléchargements non autorisés) |
| **Déduplication** | Checksum SHA256, rejet des doublons |
| **Réseau** | Écoute uniquement 127.0.0.1:8000 (localhost), firewall RPi recommandé |
| **Données** | SQLite local, aucune exposition réseau directe |

### Recommandations

1. **Token fort** : Générer via `openssl rand -hex 32`
2. **Firewall RPi** : Autoriser seulement le réseau local
3. **Rotation** : Changer le token régulièrement
4. **Ne pas commiter** : Utiliser `.env` local, jamais en Git
5. **Permissions** : Vérifier `chmod 600` sur `.env`

## Logs et debugging

### Voir les logs en direct

```bash
docker-compose logs -f shootscore-api
```

### Log levels

- `DEBUG` : Détails complets (non utilisé par défaut)
- `INFO` : Opérations normales (uploads, startups)
- `WARNING` : Doublons, erreurs non critiques
- `ERROR` : Erreurs de traitement

## Maintenance

### Arrêter le serveur

```bash
docker-compose down
```

### Redémarrer

```bash
docker-compose restart shootscore-api
```

### Nettoyer les uploads anciens

```bash
# Supprimer images de plus de 30 jours
find data/uploads -type f -mtime +30 -delete
```

### Réinitialiser la base de données

```bash
rm db/shootscore.db
docker-compose restart shootscore-api  # Recréé automatiquement
```

## Monitoring

### Dashboard improvisé

```bash
# Terminal 1: Logs
docker-compose logs -f

# Terminal 2: Compte des uploads du jour
watch -n 5 'find data/uploads/$(date +%Y-%m-%d) -type f | wc -l'

# Terminal 3: Stats base de données
watch 'sqlite3 db/shootscore.db \
  "SELECT COUNT(*) as total, MAX(uploaded_at) as last_upload FROM uploads;"'
```

## Troubleshooting

| Problème | Solution |
|----------|----------|
| `Connection refused` | `docker-compose ps` pour vérifier l'état |
| `401 Unauthorized` | Vérifier le token dans Authorization header |
| `Port already in use` | Changer port dans docker-compose.yml |
| `Permission denied` | `sudo chown -R $USER:$USER data db` |
| `Out of disk` | Nettoyer les vieux uploads |

## Déploiement en production

### Systemd auto-restart

Créer `/etc/systemd/system/shootscore-upload.service` (voir [README_RPi.md](README_RPi.md))

### Backup

```bash
# Sauvegarder les données
rsync -avz data/ /mnt/backup/shootscore/

# Sauvegarder la DB
cp db/shootscore.db /mnt/backup/shootscore.db.backup
```

## Performance

- **RPi 3** : ~100-200 uploads/jour sans problème
- **RPi 4** : >1000 uploads/jour
- **Stockage** : ~500KB par image JPEG (1280x1280)
- **CPU** : <5% CPU au repos

## Voir aussi

- [Guide d'intégration mobile](../SETUP_UPLOAD.md)
- [README RPi détaillé](README_RPi.md)
- [Service upload (mobile)](../mobile/services/uploadService.ts)
- [Config mobile](../mobile/services/uploadConfig.ts)
