# Guide d'intégration : Système d'upload automatique

## Vue d'ensemble

**Objectif** : Toutes les photos brutes prises par les utilisateurs sont automatiquement envoyées à un serveur local (Raspberry Pi) pour l'entraînement continu.

**Architecture** :
- **Mobile** : Upload asynchrone et automatique après chaque tir
- **RPi (Serveur)** : Reçoit via HTTP POST, stocke localement, déduplication via checksum
- **Docker** : Containerisation du serveur pour portabilité

## Flux complet

```
Utilisateur prend une photo
  ↓
Application détecte les impacts
  ↓
Photo brute + métadonnées enqueued en AsyncStorage
  ↓
Service d'upload essaie immédiatement (WiFi/réseau)
  ↓
Serveur RPi reçoit et calcule checksum SHA256
  ↓
Si nouveau → Stocke dans data/uploads/YYYY-MM-DD/
  ↓
Si doublon → Rejette (409 Conflict)
  ↓
App marque comme uploadée
  ↓
Retry automatique (backoff exponentiel) en cas d'échec
```

## Installation étape par étape

### Phase 1 : Configurer le serveur RPi (30 min)

1. **SSH sur le Raspberry Pi** :
   ```bash
   ssh pi@<IP-RPi>
   ```

2. **Cloner le code serveur** :
   ```bash
   git clone <repo> /home/pi/shootscore
   cd /home/pi/shootscore/server
   ```

3. **Installer Docker** (voir [server/README_RPi.md](server/README_RPi.md)) :
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

4. **Configurer le token** :
   ```bash
   cp .env.example .env
   openssl rand -hex 32  # Générer un token sécurisé
   nano .env              # Remplacer API_SECRET_TOKEN
   ```

5. **Lancer le serveur** :
   ```bash
   docker-compose build
   docker-compose up -d
   docker-compose logs -f
   ```

6. **Tester la connexion** (depuis le RPi) :
   ```bash
   curl -s http://localhost:8000/health
   ```

### Phase 2 : Configurer l'app mobile (10 min)

1. **Récupérer l'adresse IP du RPi** :
   ```bash
   ssh pi@<IP-RPi> hostname -I
   # Exemple: 192.168.1.42
   ```

2. **Éditer `mobile/services/uploadConfig.ts`** :
   ```typescript
   export const UPLOAD_CONFIG = {
     UPLOAD_SERVER_URL: 'http://100.77.3.9:8000',  // Remplacer IP
     UPLOAD_API_TOKEN: '4be63efcf7f582c8a545164ea80585273d89e176f209c6490814ddd4a9374b3b', // Token du .env du serveur
     UPLOADS_ENABLED: true,
     // ...
   };
   ```

3. **Recompiler l'app** :
   ```bash
   cd mobile
   npm run build
   # ou pour dev:
   npx expo start --clear
   ```

### Phase 3 : Tester (5 min)

1. **Prendre une photo** dans l'app
2. **Vérifier les logs** du serveur :
   ```bash
   ssh pi@<IP-RPi>
   cd /home/pi/shootscore/server
   docker-compose logs -f
   # Vous devriez voir: "Upload successful: series_X/shot_Y"
   ```

3. **Vérifier les fichiers uploadés** :
   ```bash
   ls /home/pi/shootscore/server/data/uploads/$(date +%Y-%m-%d)/
   ```

## Configuration détaillée

### Fichiers clés

| Fichier | Rôle |
|---------|------|
| `server/app/main.py` | API FastAPI (POST /api/v1/upload) |
| `server/docker-compose.yml` | Configuration Docker (port, volumes, env) |
| `server/.env` | Token d'authentification (⚠️ ne pas commiter) |
| `mobile/services/uploadService.ts` | Logique client (queue, retry, dedup) |
| `mobile/services/uploadConfig.ts` | Configuration de l'app (URL, token) |
| `mobile/services/storage.ts` | Intégration au workflow (call après cada tir) |

### Base de données (SQLite)

Le serveur maintient un index des uploads :

```bash
ssh pi@<IP-RPi>
sqlite3 /home/pi/shootscore/server/db/shootscore.db
sqlite> SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT 5;
```

### Déduplication

- Chaque photo est hashée via **SHA256**
- Le serveur rejette les doublons (code 409)
- L'app détecte et n'essaie pas de renvoyer

## Récupérer les images pour l'entraînement

### Manuelle (SSH)

```bash
# Copier localement
scp -r pi@192.168.1.42:/home/pi/shootscore/server/data/uploads ~/shootscore_images

# Ou avec rsync (plus rapide pour les mises à jour)
rsync -avz pi@192.168.1.42:/home/pi/shootscore/server/data/uploads/ ~/shootscore_images/
```

### Organisée par date

```bash
ls ~/shootscore_images/
# 2026-09-27/
# 2026-09-28/
# ...

ls ~/shootscore_images/2026-09-27/
# series_123_shot_0.jpg
# series_123_shot_1.jpg
# series_124_shot_0.jpg
```

### Avec labels (optionnel)

Les métadonnées des impacts sont stockées en base. Vous pouvez les exporter :

```python
# Script Python pour générer les labels YOLO
import sqlite3
import json

db = sqlite3.connect('/path/to/shootscore.db')
db.row_factory = sqlite3.Row

for row in db.execute('SELECT * FROM uploads'):
    metadata = json.loads(row['metadata'])
    # Générer label YOLO...
```

## Sécurité

✅ **Protégé par défaut** :
- Authentification Bearer obligatoire
- Pas d'endpoint GET (impossible de télécharger les images via l'API)
- Réseau local uniquement (127.0.0.1:8000)
- Déduplication par checksum

⚠️ **Recommandations** :
1. Régénérer le token régulièrement
2. Ne jamais commiter `.env` ni les tokens en dur
3. Firewall RPi : `sudo ufw allow from 192.168.1.0/24 to any port 8000`
4. Changer le mot de passe SSH par défaut du RPi

## Troubleshooting

### L'app refuse de se connecter

**Symptômes** :
- Logs `[Upload] Network error` dans le service

**Solutions** :
1. Vérifier l'adresse IP du RPi : `ping 192.168.1.X`
2. Vérifier que le serveur tourne : `docker-compose ps`
3. Vérifier le firewall : `sudo ufw status`
4. Tester depuis le téléphone : `curl -s http://192.168.1.X:8000/health`

### Le serveur rejette le token

**Symptômes** :
- Réponse `401 Unauthorized`

**Solutions** :
1. Vérifier que le token dans `uploadConfig.ts` correspond au `.env` du serveur
2. Régénérer un nouveau token : `openssl rand -hex 32`

### Les images n'apparaissent pas

**Symptômes** :
- Aucun fichier dans `data/uploads/`

**Solutions** :
1. Vérifier les logs : `docker-compose logs -f`
2. Vérifier les permissions : `ls -la data/uploads/`
3. Vérifier que l'app enqueued : Logs `[Upload] Queued: series_X/shot_Y`

## Monitoring

### Voir les uploads en temps réel

```bash
# Ouvrir 3 terminaux

# Terminal 1: Logs
ssh pi@192.168.1.42
cd /home/pi/shootscore/server
docker-compose logs -f

# Terminal 2: Compter les uploads du jour
watch -n 5 'find data/uploads/$(date +%Y-%m-%d) -type f | wc -l'

# Terminal 3: Consulter la base
watch 'sqlite3 db/shootscore.db "SELECT COUNT(*), MAX(uploaded_at) FROM uploads;"'
```

## Maintenance

### Nettoyer les vieux uploads

```bash
# Supprimer les images de plus de 30 jours
ssh pi@192.168.1.42
find /home/pi/shootscore/server/data/uploads -type f -mtime +30 -delete

# Vérifier l'espace disque
df -h
```

### Backup

```bash
# Sauvegarder les uploads
rsync -avz pi@192.168.1.42:/home/pi/shootscore/server/data/ ~/backups/shootscore/

# Sauvegarder la base de données
scp pi@192.168.1.42:/home/pi/shootscore/server/db/shootscore.db ~/backups/
```

## Automatisation

### Redémarrage auto au boot du RPi

Voir [server/README_RPi.md - Automatisation](server/README_RPi.md)

### Export automatique des images

```python
# Script à ajouter dans cron
# crontab -e
0 3 * * * rsync -avz /home/pi/shootscore/server/data/uploads/ /mnt/external_drive/backups/

```

## Prochaines étapes

1. ✅ Serveur running sur RPi
2. ✅ App uploading les photos
3. ⏭️ Récupérer les images : `rsync` les uploads
4. ⏭️ Générer les labels YOLO depuis les métadonnées
5. ⏭️ Entraîner continuellement avec `train_yolo.py`
6. ⏭️ Exporter les nouveaux modèles en ONNX
7. ⏭️ Mettre à jour les modèles sur le RPi et le téléphone

## Support

- Docs serveur : [server/README_RPi.md](server/README_RPi.md)
- Config mobile : [mobile/services/uploadConfig.ts](mobile/services/uploadConfig.ts)
- Service upload : [mobile/services/uploadService.ts](mobile/services/uploadService.ts)
- Intégration : [mobile/services/storage.ts](mobile/services/storage.ts)
