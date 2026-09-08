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




##### NEW #####
Parfait. Voici le mode d’emploi complet de A a Z sur l’etat actuel du projet, avec le nouveau pipeline 4 reperes.

0. Prerequis

Cloner le repo parent puis initialiser le submodule mobile.
Avoir Python 3.10+ et Node/Android SDK.
Commandes:
git submodule update --init --recursive
uv sync

Si tu n’utilises pas uv:
pip install -e .

1. Labeliser les 4 reperes "1" (top, left, bottom, right)
Script:
label_axis_markers.py

Commande:
python label_axis_markers.py data/raw

Ce que ca produit:

Un fichier voisin de chaque image: nom_image_axis_labels.json
Raccourcis utiles:

clic gauche: poser/remplacer le point selectionne
clic droit: supprimer le point le plus proche
1/2/3/4: choisir top/left/bottom/right
S: save
N/P: suivante/precedente
Q: quitter
2. Construire le dataset YOLO des reperes
Script:
prepare_axis_yolo_dataset.py

Commande:
python .\src\prepare_axis_yolo_dataset.py data/raw --out data/axis_yolo --val-ratio 0.2 --box-size 48

Ce que ca produit:

data/axis_yolo/images/train|val
data/axis_yolo/labels/train|val
data/axis_yolo/dataset.yaml
3. Entrainer le modele YOLO des reperes
Script:
train_yolo.py

Important:

Le script ecrit sous nom yolo_impacts, donc separe bien les dossiers de sortie pour ne pas ecraser les trainings.
Commande recommandee pour les reperes:
python train_yolo.py --data data/axis_yolo/dataset.yaml --model yolo11n.pt --epochs 200 --imgsz 1056 --batch 8 --out models_axis --aug-preset axis_markers --training-profile high_performance

Poids obtenus:

models_axis/yolo_impacts/weights/best.pt
4. Exporter le modele des reperes en ONNX pour Android
Script:
export_axis_yolo_onnx.py

Commande:
python .\src\export_axis_yolo_onnx.py --weights models_axis/yolo_impacts/weights/best.pt

Asset cible:

assets -> axis_markers_yolo.onnx
5. Preparer les images flat pour le dataset impacts
Tu as 2 options:

Si tu as deja des images flat annotees dans outputs, utilise-les.
Sinon genere-les avec le pipeline marker:
python pipeline_markers.py data/raw --out outputs --debug
Pipeline:
pipeline_markers.py

detection 4 reperes
homographie
flat 1056
YOLO impacts
6. Labeliser les impacts
Script:
label_impacts.py

Commande:
python label_impacts.py outputs --skip-done

Ce que ca produit:

nom_image_labels.json a cote de chaque image flat
Raccourcis:

clic gauche: ajouter impact
clic droit: supprimer impact proche
Z: undo
S: save
N/P: suivante/precedente
Q: quitter
7. Construire le dataset YOLO des impacts
Script:
prepare_yolo_dataset.py

Commande:
python prepare_yolo_dataset.py outputs --out data/yolo --val-ratio 0.2

Ce que ca produit:

data/yolo/dataset.yaml + images/labels train/val
8. Entrainer le modele YOLO impacts
Script:
train_yolo.py

Commande recommandee:
python train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt --epochs 200 --imgsz 1056 --batch 8 --out models

Poids obtenus:

models/yolo_impacts/weights/best.pt
9. Exporter le modele impacts en ONNX pour Android
Script:
export_onnx.py

Commande:
python export_onnx.py --weights models/yolo_impacts/weights/best.pt

Asset cible:

assets -> impact_yolo.onnx
10. Tester le pipeline desktop de bout en bout
Commande:
python pipeline_markers.py data/raw/photo.jpg --axis-weights models_axis/yolo_impacts/weights/best.pt --impact-weights models/yolo_impacts/weights/best.pt --show

11. Deployer l’application sur telephone Android

Activer options dev + debogage USB sur le telephone.

Brancher en USB.

Verifier:
adb devices

Build/install Android:
cd mobile/android
gradlew.bat app:compileDebugKotlin
gradlew.bat app:installDebug

Lancer l’app:
adb shell monkey -p com.rexmi.shootscore -c android.intent.category.LAUNCHER 1

Pour dev JS:
cd mobile
npm install
npx expo start

Comme tu utilises un module natif, il faut une vraie build Android (pas Expo Go).

12. Ce qui est maintenant archive

Legacy disque noir, anneaux detectes, patch-CNN, heatmap:
archive_2026-09-marker-migration
Assets mobile legacy:
archive_2026-09-marker-migration

13. Booster drastiquement la performance YOLO (meme si entrainement long)

Objectif: pousser le mAP au maximum, quitte a augmenter fortement le temps de calcul.

Points les plus impactants (ordre recommande):

- Qualite labels > quantite: relire 100% du val et corriger les labels ambigus.
- Plus de data difficile: flou, faible lumiere, angle fort, reflets, zoom variable.
- Hard negatives: ajouter des images sans cible exploitable et sans impact.
- Pas de rotation/flip pour les reperes axis: top/left/bottom/right sont des classes orientees.
- Modele plus gros: preferer yolo11l.pt ou yolo11x.pt si GPU suffisant.
- Resolution plus haute: essayer imgsz 1280 puis 1536.
- Entrainement plus long: 300 a 600 epochs, patience elevee.
- Selection par validation stricte: conserver seulement les runs qui generalisent sur un lot hors entrainement.
- Ensemble de modeles: combiner 2 a 3 meilleurs checkpoints pour inference finale.

Exemple commande reperes (qualite max):

python train_yolo.py --data data/axis_yolo/dataset.yaml --model yolo11x.pt --epochs 450 --imgsz 1280 --batch 4 --out models_axis --aug-preset axis_markers --training-profile high_performance

Exemple commande impacts (qualite max):

python train_yolo.py --data data/yolo/dataset.yaml --model yolo11x.pt --epochs 450 --imgsz 1280 --batch 4 --out models --aug-preset robust --training-profile high_performance

Options avancees si tu passes par la CLI Ultralytics directe:

```powershell
yolo detect train data=data/axis_yolo/dataset.yaml model=yolo11x.pt imgsz=1280 epochs=450 batch=4 optimizer=AdamW cos_lr=True close_mosaic=20 patience=150 cache=True
```

Anti-overfitting (a activer progressivement):

- reduire les augmentations agressives si le val diverge;
- augmenter le dataset reel avant d'augmenter encore les epochs;
- conserver un split val fixe pour comparer les runs de maniere fiable.