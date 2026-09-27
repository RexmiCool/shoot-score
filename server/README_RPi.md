# ShootScore Upload Server — Guide d'installation sur Raspberry Pi

## Architecture

- **Serveur FastAPI** : Reçoit les images brutes en HTTP POST
- **Authentification** : Token Bearer simple
- **Stockage** : Local sur le RPi, organisé par date
- **Déduplication** : Via checksum SHA256
- **Containerisation** : Docker pour isolation et portabilité

## Prérequis

- **Raspberry Pi 3/4+** (4GB RAM minimum recommandé)
- **Raspbian OS** (Debian-based)
- **Docker** et **Docker Compose** installés
- **WiFi** pour la connexion au téléphone

## Installation

### 1. Installer Docker sur le Raspberry Pi

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Ajouter votre utilisateur au groupe docker
sudo usermod -aG docker $USER
newgrp docker

# Vérifier l'installation
docker --version
docker-compose --version
```

### 2. Cloner ou copier le serveur

```bash
# Sur le RPi
git clone <votre-repo> /home/pi/shootscore
cd /home/pi/shootscore/server
```

### 3. Configurer le token d'authentification

```bash
# Créer un fichier .env sécurisé
cp .env.example .env

# Générer un token sécurisé (minimum 32 caractères)
openssl rand -hex 32

# Éditer .env et remplacer API_SECRET_TOKEN
nano .env
# Exemple:
# API_SECRET_TOKEN=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
```

### 4. Construire et lancer le serveur Docker

```bash
cd /home/pi/shootscore/server

# Construire l'image
docker-compose build

# Lancer le serveur en arrière-plan
docker-compose up -d

# Vérifier que c'est opérationnel
docker-compose logs -f

# Après quelques secondes, vous devriez voir:
# shootscore-api | INFO:     Application startup complete
```

### 5. Tester le serveur

```bash
# Récupérer l'adresse IP locale du RPi
hostname -I

# Tester le health check depuis le RPi
curl -s http://localhost:8000/health | jq

# Tester depuis le téléphone (remplacer 192.168.1.X par l'IP réelle)
curl -s http://192.168.1.X:8000/health
```

### 6. Configurer le téléphone

Sur le téléphone, dans `mobile/services/uploadService.ts` :

```typescript
// Remplacer par l'adresse IP réelle du RPi
const UPLOAD_SERVER_URL = 'http://192.168.1.X:8000';
const UPLOAD_API_TOKEN = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0'; // Token du .env
```

## Sécurité

### ✅ Ce qui est protégé

- **Pas de GET publique** : Seul POST est autorisé, les images ne peuvent pas être téléchargées via l'API
- **Authentification Bearer** : Token requis, stocké localement
- **Réseau local uniquement** : Écoute uniquement sur 127.0.0.1:8000 côté host (port 8000 en interne)
- **Déduplication** : Évite les uploads dupliqués via checksum SHA256
- **Base de données locale** : SQLite, pas d'exposition réseau

### ⚠️ Recommandations supplémentaires

1. **Firewall RPi** :
   ```bash
   # Autoriser seulement depuis le réseau WiFi local
   sudo ufw allow from 192.168.1.0/24 to any port 8000
   sudo ufw enable
   ```

2. **Token fort** :
   - Régénérer le token avec `openssl rand -hex 32`
   - Changer régulièrement
   - Ne jamais commiter en dur

3. **Réseau isolé** (optionnel) :
   - Mettre le RPi sur un réseau WiFi séparé du reste
   - Ne jamais exposer le port 8000 sur internet

## Utilisation

### Récupérer les images uploadées via SSH

```bash
# Depuis votre ordinateur
scp -r pi@192.168.1.X:/home/pi/shootscore/server/data/uploads ~/shootscore_data

# Ou avec rsync pour les mises à jour
rsync -avz pi@192.168.1.X:/home/pi/shootscore/server/data/ ~/shootscore_data/
```

### Consulter les logs

```bash
# Voir les 50 dernières lignes
docker-compose -f /home/pi/shootscore/server/docker-compose.yml logs --tail=50 shootscore-api

# Suivre en temps réel
docker-compose -f /home/pi/shootscore/server/docker-compose.yml logs -f shootscore-api
```

### Consulter la base de données SQLite

```bash
# Depuis le RPi
sqlite3 /home/pi/shootscore/server/db/shootscore.db

# Requêtes utiles
sqlite> SELECT COUNT(*) FROM uploads;
sqlite> SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT 10;
sqlite> SELECT series_id, COUNT(*) FROM uploads GROUP BY series_id;
```

### Arrêter le serveur

```bash
cd /home/pi/shootscore/server
docker-compose down
```

### Redémarrer le serveur

```bash
cd /home/pi/shootscore/server
docker-compose restart shootscore-api
```

## Récupération en cas de problème

### Le serveur ne démarre pas

```bash
# Vérifier les logs
docker-compose logs shootscore-api

# Reconstruire l'image
docker-compose build --no-cache

# Relancer
docker-compose up -d
```

### Permission denied sur les fichiers

```bash
# Fixer les permissions
sudo chown -R $USER:$USER /home/pi/shootscore/server/data
sudo chown -R $USER:$USER /home/pi/shootscore/server/db
chmod -R 755 /home/pi/shootscore/server/data
chmod -R 755 /home/pi/shootscore/server/db
```

### Nettoyer les images anciennes

```bash
# Supprimer les uploads de plus de 30 jours
find /home/pi/shootscore/server/data/uploads -type f -mtime +30 -delete
```

## Automatisation (optionnel)

Pour relancer le serveur au redémarrage du RPi :

```bash
# Créer un script systemd
sudo nano /etc/systemd/system/shootscore-upload.service
```

```ini
[Unit]
Description=ShootScore Upload Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker-compose -f /home/pi/shootscore/server/docker-compose.yml up -d
ExecStop=/usr/bin/docker-compose -f /home/pi/shootscore/server/docker-compose.yml down
WorkingDirectory=/home/pi/shootscore/server

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable shootscore-upload.service
sudo systemctl start shootscore-upload.service
sudo systemctl status shootscore-upload.service
```

## Monitoring

Pour surveiller les uploads en temps réel :

```bash
# Terminal 1 : Voir les logs
docker-compose logs -f shootscore-api

# Terminal 2 : Compter les uploads
watch -n 5 'ls -la data/uploads/$(date +%Y-%m-%d)/ | wc -l'

# Terminal 3 : Consulter la DB
watch 'sqlite3 db/shootscore.db "SELECT COUNT(*) FROM uploads;"'
```

## Troubleshooting

| Problème | Solution |
|----------|----------|
| `Connection refused` | Vérifier que le serveur tourne : `docker-compose ps` |
| `401 Unauthorized` | Vérifier le token dans uploadService.ts |
| `Duplicate image` | Normal, le serveur rejette les doublons (checksum) |
| `Out of disk space` | Nettoyer les vieilles uploads : `rm -rf data/uploads/2026-09-*` |
| `Port already in use` | Changer le port dans docker-compose.yml |

## Support

- Logs : `/home/pi/shootscore/server/` + `docker-compose logs`
- Base de données : `db/shootscore.db` (SQLite)
- Uploads : `data/uploads/YYYY-MM-DD/`
