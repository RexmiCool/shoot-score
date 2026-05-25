# shoot-score

Outil de détection automatique des impacts de balles sur des cibles de tir,
depuis une simple photo prise sur le stand.

## Memo Operationnel

Pour une vue complete du fonctionnement global + les commandes (avec quand les utiliser), voir :

- `MEMO_COMMANDES.md`

> Note: les fichiers de l'ancienne architecture (serveur + scripts legacy)
> ont ete archives dans `old/` pour garder une racine de projet plus propre.

```
Photo brute (téléphone)
        │
        ▼
  flatten_target   ──►  image recadrée 1056×1056 px, perspective corrigée
        │
        ▼
  detect_rings     ──►  anneaux calibrés + étalonnage mm/px
        │
        ▼
  detect_impacts_yolo ──►  impacts détectés + score par zone
```

---

## Prérequis

### Développement local

- Python 3.10+  
- [uv](https://github.com/astral-sh/uv) (gestionnaire de paquets)

```powershell
uv sync          # installe toutes les dépendances (torch CUDA inclus)
```

> **GPU** : le projet est configuré pour PyTorch CUDA 12.4.
> Si votre pilote est plus ancien, modifiez l'URL dans `pyproject.toml`
> (`cu124` → `cu121` ou `cu118`) puis relancez `uv sync`.

### Via Docker (recommandé pour le serveur API)

- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/) v2
- *(Optionnel)* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) pour l'accélération GPU

---

## Structure des dossiers

```
data/
  raw/              photos brutes (entrée du pipeline)
  CornerCases/      photos difficiles pour les tests
  yolo/             dataset YOLO généré par prepare_yolo_dataset.py
  ressources/       image de référence de la cible (cible.jpg)
outputs/            résultats (un sous-dossier par image)
src/                tous les scripts
runs/               poids YOLO après entraînement
```

---

## Scripts principaux

### `src/pipeline.py` — Pipeline complet ⭐

Script principal à utiliser au quotidien. Prend une photo brute (ou un
dossier de photos) et produit l'image annotée avec tous les impacts et leurs
scores, en enchaînant automatiquement les 3 étapes.

```powershell
# Une seule photo
python src/pipeline.py data/raw/ma_photo.jpg --show

# Tout un dossier
python src/pipeline.py data/raw/ --out outputs --show

# Avec des poids YOLO spécifiques
python src/pipeline.py data/raw/ --weights models/best.pt --conf 0.3
```

**Fichiers produits** dans `outputs/<nom_image>/` :

| Fichier | Description |
|---|---|
| `<nom>_flat.jpg` | Image recadrée et corrigée en perspective |
| `<nom>_rings.jpg` | Anneaux détectés (debug visuel) |
| `<nom>_rings.json` | Étalonnage mm/px |
| `<nom>_impacts_yolo.jpg` | ✅ Résultat final : impacts annotés + scores |
| `<nom>_impacts_yolo.json` | Données structurées (impacts, scores, distances) |

**Options** :

| Option | Défaut | Description |
|---|---|---|
| `--out` | `outputs` | Dossier de sortie |
| `--weights` | `runs/detect/models/…/best.pt` | Poids YOLO |
| `--conf` | `0.25` | Seuil de confiance YOLO |
| `--iou` | `0.4` | Seuil IoU (suppression des doublons) |
| `--device` | auto | `0` = GPU, `cpu` = CPU |
| `--debug` | off | Sauvegarde les images intermédiaires |
| `--show` | off | Ouvre le résultat à la fin |

---

### `src/diff_shots.py` — Différentiel entre deux séries de tir ⭐

Compare deux photos de la **même cible** (avant et après une série) et
identifie uniquement les **nouveaux impacts**, en ignorant ceux déjà présents.

```powershell
# Photos brutes
python src/diff_shots.py data/raw/serie1.jpg data/raw/serie2.jpg --show

# Images déjà aplaties (plus rapide, évite de recalculer)
python src/diff_shots.py outputs/.../shot1_flat.jpg outputs/.../shot2_flat.jpg --show

# Tolérance d'appariement personnalisée
python src/diff_shots.py avant.jpg apres.jpg --tol 10 --show
```

**Fonctionnement** : les deux images sont recalées dans le même repère
(disque noir centré en 520×520 px). Un impact "après" est considéré
**nouveau** si aucun impact "avant" n'est à moins de `--tol` mm (défaut 8mm).

**Fichiers produits** dans le dossier de l'image "après" :

| Fichier | Description |
|---|---|
| `<nom_apres>_diff.jpg` | Anciens impacts en gris pointillé, nouveaux en couleur |
| `<nom_apres>_diff.json` | `n_new`, `score_session`, `score_total`, liste des impacts |

**Options** :

| Option | Défaut | Description |
|---|---|---|
| `--out` | `outputs` | Dossier de sortie racine |
| `--tol` | `8.0` | Tolérance d'appariement avant/après (mm) |
| `--conf` | `0.25` | Seuil de confiance YOLO |
| `--show` | off | Ouvre le résultat à la fin |

---

## Scripts du pipeline interne

Ces scripts sont appelés automatiquement par `pipeline.py`, mais peuvent aussi
être utilisés indépendamment.

### `src/flatten_target.py` — Correction de perspective

Détecte le disque noir central (Ø200mm), fitte une ellipse sur son contour
et calcule la transformation affine pour corriger la perspective. Produit une
image carrée 1056×1056 px centrée sur la cible.

```powershell
python src/flatten_target.py data/raw/ma_photo.jpg --show
python src/flatten_target.py data/raw/ --out outputs --debug
```

### `src/detect_rings.py` — Calibration des anneaux

Part d'une image `*_flat.jpg`. Calcule le profil radial de gradient depuis
le centre, détecte les pics (= bords des anneaux) et les mappe aux valeurs
théoriques (100, 125, 150, 175, 200, 225, 250mm). Produit l'étalonnage
mm/px dans `*_rings.json`.

```powershell
python src/detect_rings.py outputs/flatten/ma_photo/ma_photo_flat.jpg --show
python src/detect_rings.py outputs/flatten/ --show   # traite tout le dossier
```

### `src/detect_impacts_yolo.py` — Inférence YOLO

Charge le modèle YOLOv8 entraîné, détecte les impacts sur une image
`*_flat.jpg` et score chaque impact grâce à l'étalonnage `*_rings.json`.

```powershell
python src/detect_impacts_yolo.py outputs/flatten/ma_photo/ma_photo_flat.jpg --show
python src/detect_impacts_yolo.py outputs/flatten/ --conf 0.3 --show
```

### `src/localize_target.py` — Localisation du disque noir

Brique de base appelée par `flatten_target.py`. Détecte le grand disque noir
dans l'image brute par seuillage + morphologie. Peut être lancé seul pour
diagnostiquer des problèmes de localisation.

```powershell
python src/localize_target.py data/raw/ma_photo.jpg --debug --show
```

---

## Scripts d'entraînement du modèle YOLO

À exécuter une seule fois (ou pour réentraîner avec de nouvelles données).

### `src/label_impacts.py` — Labélisation interactive

Outil de création du ground truth. Ouvre une fenêtre interactive sur une
image `*_flat.jpg` (ou `*_rings.jpg` si disponible pour voir les anneaux).

```powershell
# Une image
python src/label_impacts.py outputs/flatten/ma_photo/ma_photo_flat.jpg

# Tout un dossier (navigation N/P)
python src/label_impacts.py outputs/flatten/
python src/label_impacts.py outputs/flatten/ --skip-done  # ignore les déjà labelisées
```

**Contrôles** :

| Touche / Action | Effet |
|---|---|
| Clic gauche | Ajouter un impact |
| Clic droit | Supprimer l'impact le plus proche |
| `Z` | Annuler le dernier ajout |
| `S` | Sauvegarder |
| `N` | Image suivante (mode dossier) |
| `P` | Image précédente (mode dossier) |
| `Q` / Échap | Quitter (sauvegarde automatique) |

Sauvegarde : `<stem>_labels.json` dans le même dossier que le `_flat.jpg`.

### `src/prepare_yolo_dataset.py` — Préparation du dataset

Convertit les fichiers `*_labels.json` en format YOLO (80% train / 20% val).

```powershell
python src/prepare_yolo_dataset.py outputs/flatten
python src/prepare_yolo_dataset.py outputs/flatten --out data/yolo --val-ratio 0.2
```

Produit : `data/yolo/dataset.yaml` + arborescence `images/` et `labels/`.

### `src/train_yolo.py` — Entraînement

Fine-tune YOLOv8n sur le dataset d'impacts. Recommandé sur GPU (~10-30 min
pour 200 epochs avec ~30 images).

```powershell
python src/train_yolo.py                              # paramètres par défaut
python src/train_yolo.py --epochs 300 --batch 4      # GPU avec peu de VRAM
python src/train_yolo.py --model yolov8s.pt           # modèle plus grand
```

Poids produits : `runs/detect/models/yolo_impacts/weights/best.pt`

| Option | Défaut | Description |
|---|---|---|
| `--data` | `data/yolo/dataset.yaml` | Dataset |
| `--model` | `yolov8n.pt` | Modèle de départ |
| `--epochs` | `200` | Nombre d'epochs |
| `--batch` | `8` | Taille de batch (réduire si OOM) |
| `--imgsz` | `1056` | Résolution d'entraînement |
| `--device` | auto | `0` = GPU, `cpu` = CPU |

---

## Scripts utilitaires / expérimentaux

### `src/detect_impacts.py` — Détection morphologique (sans IA)

Approche alternative n'utilisant pas YOLO. Détecte les impacts par
transformées morphologiques (black-hat sur zone blanche, top-hat sur disque
noir). Moins robuste que YOLO mais ne nécessite pas d'entraînement.

```powershell
python src/detect_impacts.py outputs/flatten/ma_photo/ma_photo_flat.jpg --show
```

### `src/tune_impacts.py` — Optimisation des paramètres morphologiques

Grid search sur les 5 paramètres de `detect_impacts.py` (≈2000 combinaisons).
Évalue Precision / Recall / F1 sur toutes les images labelisées.

```powershell
python src/tune_impacts.py outputs/flatten
python src/tune_impacts.py outputs/flatten --top 30
```

Sauvegarde : `outputs/flatten/tune_results.json`

### `src/detect_cv.py` — Prototype initial (legacy)

Premier prototype de détection par OpenCV classique. Conservé à titre de
référence, remplacé par le pipeline actuel.

### `src/crop_only.py` — Recadrage simple (utilitaire)

Recadre une image sur la zone de la cible sans correction de perspective.
Utile pour un aperçu rapide.

---

## Application mobile (React Native + Expo)

L'application mobile communique avec le PC via une API REST (FastAPI).
Le PC et le téléphone doivent être sur le **même réseau Wi-Fi**.

### 1. Lancer le serveur API sur le PC

**Option A — Windows (script bat)**

```bat
start-server.bat
```

Affiche automatiquement les IP locales disponibles et démarre le serveur
sur `http://0.0.0.0:8000`. Utilise le venv `.venv\` créé par `uv sync`.

**Option B — Ligne de commande (tous OS)**

```powershell
# Installer les dépendances (fastapi + uvicorn)
uv sync

# Lancer le serveur
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Trouver l'IP du PC sur le réseau local
ipconfig   # chercher "Adresse IPv4" sous "Carte réseau sans fil Wi-Fi"
```

**Option C — Docker (recommandé en production)**

Voir la section [Déploiement Docker](#déploiement-docker) ci-dessous.

### 2. Installer et lancer l'application mobile

```powershell
cd mobile
npm install
npx expo start
```

Scanner le QR code avec **Expo Go** (Android/iOS) ou lancer sur émulateur.

### 3. Configurer l'URL du serveur dans l'app

Toucher ⚙️ en haut à droite → saisir `http://<IP_DU_PC>:8000` → tester.

### Endpoints API disponibles

| Méthode | URL | Description |
|---|---|---|
| `GET` | `/health` | État du serveur (`status`, `device`, `engine`, `arch`) |
| `POST` | `/process` | Photo → impacts détectés + image annotée (base64) |
| `POST` | `/diff` | Avant + après → nouveaux impacts + image annotée |

**`POST /process`** — paramètres `multipart/form-data` :

| Champ | Type | Défaut | Description |
|---|---|---|---|
| `image` | fichier | — | Photo de la cible (JPEG/PNG) |
| `hint_cx` | float | `0.5` | Coordonnée X normalisée (0–1) du centre estimé |
| `hint_cy` | float | `0.5` | Coordonnée Y normalisée (0–1) du centre estimé |

**`POST /diff`** — mêmes champs optionnels `hint_cx`/`hint_cy`, plus :

| Champ | Type | Description |
|---|---|---|
| `before` | fichier | Photo avant la série |
| `after` | fichier | Photo après la série |

Les réponses incluent un champ `engine` indiquant le moteur utilisé
(`"yolo"`, `"onnx"`, ou `"morpho"`).

### Sélection automatique du moteur de détection

L'API sélectionne automatiquement le meilleur moteur disponible au démarrage :

| Moteur | Condition | Accélération |
|---|---|---|
| `yolo` | Ultralytics installé + poids `best.pt` présents | GPU CUDA ou CPU |
| `onnx` | Poids `.onnx` présents (sans PyTorch) | CPU (ONNX Runtime) |
| `morpho` | Aucun poids trouvé | CPU (OpenCV uniquement) |

### Structure de l'app mobile (`mobile/`)

```
mobile/
  app/
    _layout.tsx      Navigation racine
    index.tsx        Écran d'accueil (analyser / comparer)
    result.tsx       Résultat d'une cible (impacts + score)
    diff.tsx         Résultat différentiel (nouveaux impacts)
  components/
    ImpactImage.tsx  Image annotée zoomable
    ScoreBoard.tsx   Tableau des scores par impact
    ServerConfig.tsx Modal de configuration de l'URL
  services/
    api.ts           Appels API (processImage, diffImages)
    storage.ts       Persistance locale (URL serveur)
  constants/
    Colors.ts        Palette de couleurs
```

---

## Déploiement Docker

L'API peut être lancée dans un conteneur sans installer Python ni les
dépendances sur l'hôte. L'image est construite en **deux stages** pour
minimiser la taille finale :

1. **builder** (`python:3.13-slim`) — installe les dépendances via `uv sync --frozen` dans `/app/.venv`
2. **runtime** (`python:3.13-slim`) — copie le venv + code source ; ajoute uniquement les librairies système nécessaires (`libglib2.0`, `libgl1`, `libgomp1`)

### Construction et démarrage rapide

```powershell
# Construire l'image et démarrer le conteneur en arrière-plan
docker compose up --build -d

# Suivre les logs de démarrage
docker compose logs -f

# Vérifier que l'API répond
curl http://localhost:8000/health

# Arrêter et supprimer le conteneur
docker compose down
```

### Volumes montés

| Chemin hôte | Chemin conteneur | Mode | Description |
|---|---|---|---|
| `./runs` | `/app/runs` | lecture seule | Poids YOLO (`best.pt` dans `runs/detect/models/yolo_impacts/weights/`) |
| `./outputs` | `/app/outputs` | lecture/écriture | Résultats annotés, persistés entre les redémarrages |

> Les dossiers `outputs/api/` et `runs/detect/models/yolo_impacts/weights/`
> sont créés vides dans l'image ; le conteneur démarre même si les volumes
> ne sont pas montés (fallback vers le moteur morphologique).

### Accélération GPU (NVIDIA)

Le `docker-compose.yml` réserve automatiquement **1 GPU NVIDIA**
(nécessite le [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).
Pour fonctionner en **mode CPU uniquement**, supprimer le bloc `deploy` :

```yaml
# Retirer dans docker-compose.yml pour désactiver le GPU :
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Healthcheck intégré

Docker vérifie la santé du conteneur toutes les **30 secondes** en appelant
`/health`. L'état est visible dans `docker ps` (colonne `STATUS`) :

```
CONTAINER ID   IMAGE                  STATUS
abc123def456   shoot-score-api:latest Up 2 minutes (healthy)
```

### Rebuilder après modification du code

```powershell
# Reconstruire uniquement l'image (sans redémarrer les autres services)
docker compose build api

# Redémarrer avec la nouvelle image
docker compose up -d --no-deps api
```

---

## Déploiement sur Raspberry Pi

Le Raspberry Pi utilise une architecture **ARM64 (aarch64)**. L'image Docker
est multi-plateforme mais doit être construite pour ARM64.

> **Moteur de détection sur RPi** : PyTorch/CUDA n'est pas disponible sur ARM.
> L'API bascule automatiquement sur **ONNX Runtime** si un fichier `.onnx` est
> présent dans `runs/detect/models/yolo_impacts/weights/`, sinon sur la
> **détection morphologique** (pas de poids nécessaire).
>
> Pour exporter les poids entraînés en ONNX depuis le PC :
> ```powershell
> python src/export_onnx.py
> # → runs/detect/models/yolo_impacts/weights/best.onnx
> ```

---

### Méthode A — Cross-build sur le PC (recommandée)

Construire l'image ARM64 sur votre PC Windows, l'exporter comme fichier,
puis la transférer sur le RPi. **La construction est faite sur votre PC,
pas sur le RPi — beaucoup plus rapide.**

#### 1. Construire l'image ARM64 sur le PC

Le driver `docker` par défaut ne supporte pas les exports cross-plateforme.
Il faut d'abord créer un builder avec le driver `docker-container` :

```powershell
# À faire une seule fois (crée un builder persistant nommé "rpi-builder")
docker buildx create --name rpi-builder --driver docker-container --use

# Construire pour ARM64 et exporter dans un fichier tar
docker buildx build --platform linux/arm64 `
    -t shoot-score-api:rpi `
    --output "type=docker,dest=shoot-score-api-rpi.tar" `
    .
```

> La première build prend ~15-30 min (émulation QEMU pour ARM64 +
> téléchargement des dépendances). Les suivantes sont plus rapides
> grâce au cache de couches de BuildKit.
>
> Pour vérifier que `rpi-builder` est actif : `docker buildx ls`
> (une `*` indique le builder courant).
> Pour revenir au builder par défaut après : `docker buildx use default`

#### 2. Transférer l'image sur le RPi

```powershell
# Remplacer <IP_RPI> par l'adresse IP de votre Raspberry Pi
scp shoot-score-api-rpi.tar pi@<IP_RPI>:~/
scp docker-compose.rpi.yml  pi@<IP_RPI>:~/shoot-score/
```

#### 3. Charger et démarrer sur le RPi

```bash
# Sur le Raspberry Pi (SSH)
docker load -i ~/shoot-score-api-rpi.tar

# Créer les dossiers de volumes si besoin
mkdir -p ~/shoot-score/runs/detect/models/yolo_impacts/weights
mkdir -p ~/shoot-score/outputs

# Copier les poids ONNX (optionnel, mais recommandé)
# scp depuis le PC : scp best.onnx pi@<IP>:~/shoot-score/runs/detect/models/yolo_impacts/weights/

cd ~/shoot-score
docker compose -f docker-compose.rpi.yml up -d

# Vérifier que l'API répond
curl http://localhost:8000/health
```

---

### Méthode B — Build directement sur le RPi (plus simple)

Si vous préférez ne pas utiliser buildx, clonez le dépôt directement sur
le RPi et laissez Docker construire l'image sur place. **Prévoir 30-40 min
la première fois.**

```bash
# Sur le Raspberry Pi (SSH)
git clone <URL_DU_REPO> ~/shoot-score
cd ~/shoot-score

# (Optionnel) Copier les poids ONNX depuis le PC
# scp <PC>:runs/detect/models/yolo_impacts/weights/best.onnx \
#     runs/detect/models/yolo_impacts/weights/

docker compose -f docker-compose.rpi.yml up --build -d
docker compose -f docker-compose.rpi.yml logs -f
```

---

### Configurer l'app mobile pour pointer vers le RPi

Dans l'application mobile, toucher ⚙️ → saisir `http://<IP_RPI>:8000`.

Pour connaître l'IP du RPi :

```bash
hostname -I   # sur le RPi
```

---

## Workflow complet (première utilisation)

```powershell
# 1. Installer les dépendances
uv sync

# 2. Labeliser les images (une fois)
python src/label_impacts.py outputs/flatten/ --skip-done

# 3. Préparer le dataset YOLO
python src/prepare_yolo_dataset.py outputs/flatten

# 4. Entraîner le modèle (sur GPU de préférence)
python src/train_yolo.py --epochs 200

# 5. Utiliser le pipeline sur de nouvelles photos
python src/pipeline.py data/raw/ --out outputs --show

# 6. Comparer deux séries de tir
python src/diff_shots.py data/raw/serie1.jpg data/raw/serie2.jpg --show
```

---

## Géométrie de la cible

La cible utilisée est une cible standard type pistolet 25m.

| Zone | Distance au centre | Score |
|---|---|---|
| Disque noir zone 10 | 0 – 25 mm | 10 |
| Zone 9 | 25 – 50 mm | 9 |
| Zone 8 | 50 – 75 mm | 8 |
| Zone 7 (bord disque) | 75 – 100 mm | 7 |
| Zone 6 | 100 – 125 mm | 6 |
| Zone 5 | 125 – 150 mm | 5 |
| Zone 4 | 150 – 175 mm | 4 |
| Zone 3 | 175 – 200 mm | 3 |
| Zone 2 | 200 – 225 mm | 2 |
| Zone 1 | 225 – 250 mm | 1 |

Image de sortie : **1056 × 1056 px**, ≈ **0.523 mm/px**, disque noir ≈ 191 px de rayon.
