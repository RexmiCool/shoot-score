# Memo commandes ShootScore

Le traitement de production est local et embarque dans Android. Le PC sert a traiter des images, entrainer YOLO et exporter le modele ONNX vers l'application.

## 1. Installer l'environnement Python

```powershell
uv sync
```

Le projet utilise `pyproject.toml` et `uv.lock`. L'environnement local se trouve dans `.venv`.

## 2. Traiter une image sur le PC

Pipeline complet : localisation du disque, correction de perspective, calibration des anneaux, detection YOLO et scoring.

```powershell
.\.venv\Scripts\python src/pipeline.py data/raw/photo.jpg --show
.\.venv\Scripts\python src/pipeline.py data/raw --out outputs --show
```

Le code d'orchestration est [src/pipeline.py](src/pipeline.py). Les etapes appelees sont `localize_target.py`, `flatten_target.py`, `detect_rings.py` et `detect_impacts_yolo.py`.

## 3. Comparer deux photos

```powershell
.\.venv\Scripts\python src/diff_shots.py avant.jpg apres.jpg --tol 8 --show
```

Cette commande est le comparateur desktop. Le comparateur utilise dans l'application est celui de `ImpactEngineModule.kt`, avec une tolerance native de 15 mm.

## 4. Entrainer et exporter YOLO

Preparer le dataset a partir des labels :

```powershell
.\.venv\Scripts\python src/prepare_yolo_dataset.py outputs/flatten --out data/yolo --val-ratio 0.2
```

Entrainer :

```powershell
.\.venv\Scripts\python src/train_yolo.py --data data/yolo/dataset.yaml --epochs 200 --batch 8 --imgsz 1056
```

Exporter pour Android :

```powershell
.\.venv\Scripts\python src/export_onnx.py
```

L'export doit produire ou mettre a jour :

```text
mobile/android/app/src/main/assets/impact_yolo.onnx
```

Il faut ensuite reconstruire l'APK. Les scripts heatmap et patch CNN sont experimentaux et ne sont pas charges par le chemin principal Android.

## 5. Developper l'interface mobile

```powershell
Push-Location mobile
npm install
npx expo start
Pop-Location
```

Expo Go peut etre utilise pour l'interface, mais pas pour l'inference native : le module `ImpactEngine` necessite une build Android personnalisee.

## 6. Compiler et installer Android

Verifier le code Kotlin :

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:compileDebugKotlin
Pop-Location
```

Installer une version autonome :

```powershell
Push-Location .\mobile\android
.\gradlew.bat app:installRelease
Pop-Location
adb devices
adb shell monkey -p com.rexmi.shootscore -c android.intent.category.LAUNCHER 1
```

## 7. Entrainer les CNN des quatre reperes

Annoter les photos :

```powershell
.\.venv\Scripts\python src/label_axis_markers.py data/raw
```

Preparer les ROI et entrainer les quatre positions :

```powershell
.\.venv\Scripts\python src/prepare_axis_dataset.py data/raw --out data/axis_markers --screen-aspect 0.5625
.\.venv\Scripts\python src/train_axis_cnn.py --data data/axis_markers --epochs 80
```

Exporter vers les assets Android :

```powershell
.\.venv\Scripts\python src/export_axis_markers_onnx.py
```

Reinstaller ensuite l'APK pour charger les nouveaux fichiers ONNX.

## 8. Suivre le flux d'analyse

Le flux applicatif est :

```text
camera.tsx
  -> services/api.ts
  -> src/native/ImpactEngine.ts
  -> ImpactEngineModule.kt
  -> processImage() ou diffImages()
  -> services/storage.ts
  -> diff.tsx
```

Responsabilites :

- `camera.tsx` capture, importe et recadre ;
- `api.ts` verifie Android et normalise la reponse native ;
- `ImpactEngine.ts` resout le module React Native ;
- `ImpactEngineModule.kt` execute l'analyse ;
- `storage.ts` persiste images et metadonnees ;
- `diff.tsx` affiche et permet de corriger les resultats.

## 9. Diagnostic du moteur natif

Ouvrir `debug-pipeline` avec :

```text
step=disk
step=flatten
step=impacts-flat
step=impacts-full
```

Le parametre `uri` doit pointer vers une image locale `file://...`. Les methodes appelees sont `debugDetectDisk`, `debugFlatten`, `debugDetectImpactsFlat` et `debugDetectImpactsFull` dans `ImpactEngineModule.kt`.
