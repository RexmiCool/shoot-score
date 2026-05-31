# Memo Operatoire Shoot-Score

Ce document explique:
- le fonctionnement global du projet,
- quelles commandes utiliser,
- dans quel contexte les utiliser.

Il est pense pour un usage quotidien (dev, entrainement, debug, mobile).

---

## 1) Fonctionnement Global

### 1.1 Pipeline metier (photo -> impacts)

Entree:
- Photo brute de la cible (telephone ou fichier local).

Etapes:
1. `flatten_target`:
   - detecte le disque noir central,
   - corrige la perspective,
   - produit une image standardisee `1056x1056`.
2. `detect_rings`:
   - calibre l'echelle `mm/px`,
   - retrouve les anneaux pour calculer le score.
3. Detection des impacts:
   - desktop historique: YOLO via scripts Python,
   - mobile embarque actuel: YOLO ONNX dans le module natif Android.
4. Scoring:
   - chaque impact est converti en distance au centre,
   - score de 10 a 0 selon la zone.

Sortie:
- image annotee,
- JSON structure (impacts, score, metadonnees).

### 1.2 Moteur mobile actuel

L'application Android execute localement:
- preprocessing image,
- inference YOLO ONNX,
- post-traitement (seuil confiance + NMS),
- calcul de score.

Model asset utilise sur mobile:
- `mobile/android/app/src/main/assets/impact_yolo.onnx`

Module natif principal:
- `mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEngineModule.kt`

---

## 2) Commandes Essentielles (Quoi / Quand)

## 2.1 Setup environnement

Quand:
- premier clonage,
- mise a jour des dependances Python.

Commande:

```powershell
uv sync
```

Effet:
- installe les dependances du projet dans `.venv`.

---

## 2.2 Utilisation pipeline desktop

Quand:
- traiter une ou plusieurs photos localement depuis le PC.

Commande (une image):

```powershell
.\.venv\Scripts\python src/pipeline.py data/raw/ma_photo.jpg --show
```

Commande (dossier):

```powershell
.\.venv\Scripts\python src/pipeline.py data/raw --out outputs --show
```

Effet:
- enchaine flatten + rings + detection,
- ecrit les resultats dans `outputs/...`.

---

## 2.3 Comparaison avant/apres serie

Quand:
- identifier uniquement les nouveaux impacts entre deux tirs.

Commande:

```powershell
.\.venv\Scripts\python src/diff_shots.py avant.jpg apres.jpg --tol 8 --show
```

Effet:
- compare les impacts,
- renvoie `n_new`, `score_session`, `score_total`.

---

## 2.4 Entrainement YOLO (desktop)

Quand:
- ameliorer le modele de detection d'impacts.

A. preparer dataset YOLO:

```powershell
.\.venv\Scripts\python src/prepare_yolo_dataset.py outputs/flatten --out data/yolo --val-ratio 0.2
```

B. entrainer:

```powershell
.\.venv\Scripts\python src/train_yolo.py --data data/yolo/dataset.yaml --epochs 200 --batch 8 --imgsz 1056
```

Effet:
- produit un nouveau checkpoint `best.pt` dans `runs/detect/models/yolo_impacts/weights/`.

---

## 2.5 Export YOLO pour mobile (obligatoire apres nouvel entrainement)

Quand:
- chaque fois que `best.pt` change,
- avant de rebuild l'app Android.

Commande:

```powershell
.\.venv\Scripts\python src/export_onnx.py
```

Effet:
- exporte `best.pt` en ONNX,
- copie automatiquement l'asset Android vers:
  - `mobile/android/app/src/main/assets/impact_yolo.onnx`.

---

## 2.6 Build et installation Android standalone (sans Metro)

Quand:
- apres modification du code natif Android,
- apres changement du modele ONNX mobile,
- quand on veut tester l'app sans PC ni serveur Metro.

A. compiler Kotlin (verification rapide):

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:compileDebugKotlin
Pop-Location
```

B. installer la version standalone sur telephone connecte:

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:installRelease
Pop-Location
```

C. lancer l'application:

```powershell
adb shell monkey -p com.rexmi.shootscore -c android.intent.category.LAUNCHER 1
```

Notes:
- `installRelease` embarque le bundle JS dans l'APK: l'application fonctionne sans Metro.
- `installDebug` reste utile pour le developpement UI avec Metro (hot reload).

---

## 2.7 Verification appareil Android (ADB)

Quand:
- avant un deploy,
- si l'installation ne demarre pas.

Commande:

```powershell
adb devices
```

Effet attendu:
- un appareil en statut `device`.

---

## 2.8 Lancement serveur sans Docker (dev / test rapide)

Quand:
- tester le mode serveur sans Docker.

Commande:

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\uvicorn api:app --app-dir old\server --host 0.0.0.0 --port 8000
```

Afficher l'IP locale pour la configurer dans l'app:

```powershell
ipconfig | Select-String "IPv4"
```

---

## 2.9 Lancement serveur avec Docker

Quand:
- deployer le serveur sur un PC ou un Raspberry Pi,
- vouloir un environnement isole et reproductible,
- utiliser le mode hybride serveur + embarque de l'app mobile.

Pre-requis : Docker Desktop installe et demarre.

### A. PC (x86, avec ou sans GPU NVIDIA)

1. Preparer l'API dans `src/` (le Dockerfile copie `src/`) :

```powershell
Copy-Item old\server\api.py src\api.py
```

2. Construire l'image depuis la racine du projet :

```powershell
docker build -f old\server\Dockerfile -t shoot-score-api:latest .
```

3. Lancer le container (avec GPU NVIDIA si disponible) :

```powershell
docker run -d `
  --name shoot-score-api `
  -p 8000:8000 `
  --gpus all `
  -v ${PWD}\runs:/app/runs:ro `
  -v ${PWD}\outputs:/app/outputs `
  shoot-score-api:latest
```

Sans GPU (CPU uniquement) : retirer `--gpus all`.

Alternative avec docker compose :

```powershell
# Depuis la racine du projet (pas depuis old/server/)
docker compose -f old\server\docker-compose.yml `
  --project-directory . `
  up -d --build
```

4. Verifier que le serveur repond :

```powershell
curl http://localhost:8000/health
```

5. Arreter le container :

```powershell
docker stop shoot-score-api
docker rm shoot-score-api
```

### B. Raspberry Pi (ARM64, sans GPU)

Sur le Raspberry Pi, depuis la racine du projet clone :

```bash
cp old/server/api.py src/api.py
docker compose -f old/server/docker-compose.rpi.yml --project-directory . up -d --build
curl http://localhost:8000/health
```

Notes:
- Le moteur bascule automatiquement sur ONNX Runtime (CPU) si `best.pt` absent.
- Exporter d'abord le modele au format ONNX sur le PC avant de copier sur le RPi :

```powershell
.\venv\Scripts\python src/export_onnx.py
```

- Monter `runs/detect/models/yolo_impacts/weights/best.onnx` dans le container RPi.

---

## 2.10 Demarrage app mobile (dev JS)

Quand:
- developpement interface mobile Expo.

Commande:

```powershell
cd mobile
npx expo start --tunnel --clear
```

Note:
- le mode inference native requiert une build Android installee (pas Expo Go seul).
- ce mode depend de Metro et sert au dev UI, pas au test standalone.

---

## 3) Commandes De Debug Utiles

## 3.1 Tester uniquement flatten

Quand:
- la cible est mal recadree.

```powershell
.\.venv\Scripts\python src/flatten_target.py data/raw/ma_photo.jpg --debug --show
```

## 3.2 Tester uniquement anneaux

Quand:
- le score semble incoherent (calibration mm/px suspecte).

```powershell
.\.venv\Scripts\python src/detect_rings.py outputs/.../ma_photo_flat.jpg --show
```

## 3.3 Tester detection YOLO desktop

Quand:
- comparer mobile vs desktop.

```powershell
.\.venv\Scripts\python src/detect_impacts_yolo.py outputs/.../ma_photo_flat.jpg --conf 0.25 --iou 0.4 --show
```

---

## 4) Scenarios Recommandes

## Scenario A - Je viens de re-entrainer YOLO, je veux le phone a jour

1. Export modele mobile:

```powershell
.\.venv\Scripts\python src/export_onnx.py
```

2. Reinstaller app standalone:

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:installRelease
Pop-Location
```

3. Lancer app:

```powershell
adb shell monkey -p com.rexmi.shootscore -c android.intent.category.LAUNCHER 1
```

## Scenario C - Je veux tester le mode serveur depuis l'app

1. (Si pas deja fait) Preparer `src/api.py` :

```powershell
Copy-Item old\server\api.py src\api.py
```

2. Lancer le serveur Docker :

```powershell
docker run -d --name shoot-score-api -p 8000:8000 `
  -v ${PWD}\runs:/app/runs:ro `
  -v ${PWD}\outputs:/app/outputs `
  shoot-score-api:latest
```

Ou sans Docker :

```powershell
$env:PYTHONPATH = "src"
.\venv\Scripts\uvicorn api:app --app-dir old\server --host 0.0.0.0 --port 8000
```

3. Recuperer l'IP du PC sur le reseau Wi-Fi :

```powershell
ipconfig | Select-String "IPv4"
```

4. Dans l'app mobile : **Parametres** -> entrer `http://<IP>:8000` -> **Tester** -> **Enregistrer**.

5. Prendre une photo : le champ `engine` doit afficher `yolo` ou `onnx` (pas `offline-native`).

6. Pour tester le fallback embarque : couper le serveur puis reprendre une photo.

---

## Scenario B - L'app mobile retourne une erreur d'inference

1. Verifier build compile:

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:compileDebugKotlin
Pop-Location
```

2. Reinstaller app standalone:

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:installRelease
Pop-Location
```

3. Verifier appareil:

```powershell
adb devices
```

---

## 5) Fichiers Clefs

- API serveur: `old/server/api.py` (copier dans `src/api.py` pour Docker)
- Pipeline desktop: `src/pipeline.py`
- Detection YOLO desktop: `src/detect_impacts_yolo.py`
- Export modele mobile: `src/export_onnx.py`
- Moteur natif Android: `mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEngineModule.kt`
- Asset modele mobile: `mobile/android/app/src/main/assets/impact_yolo.onnx`
- Config dependances Android: `mobile/android/app/build.gradle`

---

## 6) Rappels Importants

- Toujours utiliser `\.venv\Scripts\python` pour les scripts projet.
- Apres un nouvel entrainement YOLO, il faut refaire l'export ONNX mobile.
- Expo Go seul ne suffit pas pour l'inference native: il faut une build Android installee.
- Pour tester sans Metro, utiliser la build `installRelease`.
