# ShootScore

ShootScore est une application Android de scoring de cible C50.
Le pipeline de production est 100% local (pas de serveur) et repose maintenant uniquement sur les quatre reperes "1":

- top
- left
- bottom
- right

Le disque noir central et la detection des anneaux ne sont plus utilises dans le flux principal.

## Architecture cible (active)

```text
Photo brute
  -> Detection auto des 4 reperes (YOLO axis markers)
  -> Pre-placement UI des 4 points
  -> Correction manuelle utilisateur
  -> Homographie
  -> Image flat 1056x1056
  -> Detection impacts (YOLO)
  -> Scoring geometrique theorique
```

## Source de verite geometrique

La geometrie est definie uniquement par les quatre reperes:

- top
- left
- bottom
- right

A partir de ces points:

1. homographie vers un modele C50 ideal 1056x1056
2. centre connu par construction: `(528, 528)`
3. anneaux de score generes theoriquement (pas detectes)

## Structure du projet

```text
mobile/ (submodule)
  app/                           ecrans React Native
  services/                      contrats JS/TS vers moteur natif
  src/native/ImpactEngine.ts     wrapper module Android
  android/app/src/main/java/...  moteur Kotlin
  android/app/src/main/assets/   modeles embarques (actif: impact_yolo.onnx)

src/
  pipeline_markers.py            pipeline desktop principal (4 reperes)
  detect_axis_markers.py         detection YOLO des 4 reperes
  flatten_markers.py             homographie + modele geometrique C50
  detect_impacts_yolo.py         detection YOLO des impacts + scoring
  train_yolo.py                  entrainement YOLO
  prepare_axis_yolo_dataset.py   dataset YOLO des 4 reperes
  yolo_to_axis_labels.py         conversion predictions -> labels axes
  prepare_yolo_dataset.py        dataset YOLO impacts
  label_axis_markers.py          annotation des 4 reperes
  label_impacts.py               annotation des impacts
  export_onnx.py                 export ONNX impacts

old/archive_2026-09-marker-migration/
  scripts et modeles legacy archives
```

## Pipeline mobile (actif)

Fichiers principaux:

- `mobile/app/camera.tsx`
- `mobile/app/diff.tsx`
- `mobile/services/api.ts`
- `mobile/src/native/ImpactEngine.ts`
- `mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEngineModule.kt`

Comportement:

1. la photo passe par le placement des 4 reperes
2. l'utilisateur peut corriger les points
3. la rectification est calculee via homographie
4. la detection d'impacts est executee sur l'image flat
5. le score est calcule geometriquement

## Modeles Android

Actif:

- `mobile/android/app/src/main/assets/impact_yolo.onnx`

Attendu pour la detection auto des reperes (a ajouter):

- `mobile/android/app/src/main/assets/axis_markers_yolo.onnx`

Archives (deplaces):

- `mobile/docs/archive_2026-09-marker-migration/assets/axis_marker_*.onnx`
- `mobile/docs/archive_2026-09-marker-migration/assets/impact_heatmap_cnn.tflite`
- `mobile/docs/archive_2026-09-marker-migration/assets/mini_impact_cnn.tflite`

## Pipeline Python (actif)

Execution image unique:

```powershell
python src/pipeline_markers.py data/raw/photo.jpg --show
```

Execution dossier:

```powershell
python src/pipeline_markers.py data/raw --out outputs --debug
```

## Entrainement YOLO reperes

Preparation dataset depuis labels JSON:

```powershell
python src/prepare_axis_yolo_dataset.py data/raw --out data/axis_yolo --val-ratio 0.2
```

Entrainement:

```powershell
python src/train_yolo.py --data data/axis_yolo/dataset.yaml --model yolo11n.pt --epochs 200 --imgsz 1056
```

## Entrainement YOLO impacts

Preparation dataset:

```powershell
python src/prepare_yolo_dataset.py outputs --out data/yolo --val-ratio 0.2
```

Entrainement:

```powershell
python src/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt --epochs 200 --imgsz 1056
```

Export ONNX:

```powershell
python src/export_onnx.py
```

## Developpement mobile

```powershell
Push-Location mobile
npm install
npx expo start
Pop-Location
```

Build Android:

```powershell
Push-Location mobile\android
.\gradlew.bat app:compileDebugKotlin
.\gradlew.bat app:installRelease
Pop-Location
```

## Nettoyage effectue

Les elements obsoletes (disque noir, anneaux detectes, patch-CNN, heatmap) ont ete retires du chemin actif et archives dans:

- `old/archive_2026-09-marker-migration/`
- `mobile/docs/archive_2026-09-marker-migration/`

Ce nettoyage permet de maintenir une base code claire autour de la seule architecture cible: 4 reperes -> homographie -> impacts YOLO -> scoring geometrique.
