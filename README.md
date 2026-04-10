# shoot-score

Outil de détection automatique des impacts de balles sur des cibles de tir,
depuis une simple photo prise sur le stand.

```
Photo brute (téléphone)
        │
        ▼
  flatten_target   ──►  image recadrée 1040×1040 px, perspective corrigée
        │
        ▼
  detect_rings     ──►  anneaux calibrés + étalonnage mm/px
        │
        ▼
  detect_impacts_yolo ──►  impacts détectés + score par zone
```

---

## Prérequis

- Python 3.10+  
- [uv](https://github.com/astral-sh/uv) (gestionnaire de paquets)

```powershell
uv sync          # installe toutes les dépendances (torch CUDA inclus)
```

> **GPU** : le projet est configuré pour PyTorch CUDA 12.4.
> Si votre pilote est plus ancien, modifiez l'URL dans `pyproject.toml`
> (`cu124` → `cu121` ou `cu118`) puis relancez `uv sync`.

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
image carrée 1040×1040 px centrée sur la cible.

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
| `--imgsz` | `640` | Résolution d'entraînement |
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

```powershell
# Installer les dépendances (fastapi + uvicorn)
uv sync

# Lancer le serveur (remplacer l'adresse si besoin)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Trouver l'IP du PC sur le réseau local
ipconfig   # chercher "Adresse IPv4" sous "Carte réseau sans fil Wi-Fi"
```

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
| `GET` | `/health` | Vérification du serveur |
| `POST` | `/process` | Photo → impacts + image annotée |
| `POST` | `/diff` | Avant + après → nouveaux impacts |

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

Image de sortie : **1040 × 1040 px**, ≈ **0.523 mm/px**, disque noir ≈ 191 px de rayon.
