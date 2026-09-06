# ShootScore

ShootScore est une application Android qui photographie une cible, detecte les impacts et calcule les scores. Le traitement de production est entierement embarque dans l'appareil : aucune API distante ni connexion reseau n'est necessaire pendant l'analyse.

## Sommaire

- [Architecture](#architecture)
- [Flux complet d'une photo](#flux-complet-dune-photo)
- [Parcours utilisateur](#parcours-utilisateur)
- [Moteur natif Android](#moteur-natif-android)
- [Series, comparaison et stockage](#series-comparaison-et-stockage)
- [Ecrans et composants](#ecrans-et-composants)
- [Pipeline Python](#pipeline-python)
- [Entrainement et export du modele](#entrainement-et-export-du-modele)
- [CNN des quatre reperes](#cnn-des-quatre-reperes)
- [Installation et commandes](#installation-et-commandes)
- [Diagnostic](#diagnostic)

## Architecture

```text
mobile/
  app/                         Ecrans et orchestration du parcours utilisateur
  components/                  Overlays, images zoomables et tableaux de scores
  services/api.ts              Contrat JS vers le moteur natif
  services/impactEngine.ts     Couche de compatibilite vers le wrapper natif
  services/storage.ts          Historique et images persistantes
  services/uiSettings.ts       Preferences de recadrage et d'affichage
  src/native/ImpactEngine.ts   Wrapper React Native du module Android
  android/app/src/main/...     Module Kotlin et enregistrement React Native
  android/app/src/main/assets/ Modeles embarques

src/                           Pipeline Python desktop et outils ML
models/, tf_model/             Checkpoints et exports de recherche
mobile/android/.../assets/     Modeles effectivement charges par Android
data/                          Photos, annotations et datasets
outputs/                       Resultats des traitements Python
runs/                          Sorties d'entrainement YOLO
```

Le point d'entree JavaScript est [mobile/index.js](mobile/index.js), puis Expo Router charge [mobile/app/_layout.tsx](mobile/app/_layout.tsx). Le module natif est enregistre par [ImpactEnginePackage.kt](mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEnginePackage.kt) et expose le nom `ImpactEngine`.

Les fichiers les plus importants sont :

| Responsabilite | Fichier | Ce qu'il gere |
|---|---|---|
| Camera et orchestration | [mobile/app/camera.tsx](mobile/app/camera.tsx) | Acquisition, galerie, recadrage, lancement analyse/diff |
| Contrat d'analyse | [mobile/services/api.ts](mobile/services/api.ts) | Appels natifs `processImage`, `diffImages`, `rectifyPerspective` |
| Bridge JS/Android | [mobile/src/native/ImpactEngine.ts](mobile/src/native/ImpactEngine.ts) | Resolution du module natif et controle des methodes |
| Traitement image | [ImpactEngineModule.kt](mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEngineModule.kt) | Decodage, disque, flatten, YOLO, scoring, diff |
| Persistance | [mobile/services/storage.ts](mobile/services/storage.ts) | Series, tirs, images flat, corrections, export ZIP |
| Preferences | [mobile/services/uiSettings.ts](mobile/services/uiSettings.ts) | Mode crop, mode zones, mode base |
| Affichage serie | [mobile/app/diff.tsx](mobile/app/diff.tsx) | Galerie des tirs, overlays, zones, corrections manuelles |

## Flux complet d'une photo

### 1. Acquisition

L'utilisateur ouvre la camera depuis [mobile/app/index.tsx](mobile/app/index.tsx). La navigation est definie par [mobile/app/_layout.tsx](mobile/app/_layout.tsx).

Dans [mobile/app/camera.tsx](mobile/app/camera.tsx) :

- `handleCapture()` appelle `CameraView.takePictureAsync()`.
- `handleGallery()` appelle `ImagePicker.launchImageLibraryAsync()`.
- En mode `new_shot`, plusieurs photos peuvent etre selectionnees et traitees dans l'ordre.
- En mode `add_shot`, une seule photo est ajoutee a la serie existante.

La photo originale est conservee comme `photo_uri` pour eviter qu'une comparaison future ne reutilise une image deja aplatie ou recadree.

### 2. Preparation et recadrage

Toujours dans `camera.tsx` :

- `prepareImageForAnalysis()` recupere les dimensions si necessaire.
- `buildViewfinderSquareCrop()` calcule le crop correspondant au viseur camera.
- `buildCenteredSquareCrop()` calcule un crop carre centre pour les images de galerie.
- `cropToAnalysisSquare()` applique le crop avec `expo-image-manipulator`.

Le recadrage automatique est utilise avant l'analyse, sauf si l'image a deja ete transformee par un mode manuel.

Trois outils manuels sont disponibles :

- `CropModal` : zoom tactile, deplacement et selection d'un carre.
- `CornerCropModal` : selection de quatre coins puis correction de perspective.
- `AxisCropModal` : placement des points haut, gauche, bas et droite pour construire une rectification autour de la cible.

Les fonctions `applyManualCrop()`, `applyManualCornerRectification()` et `applyManualAxesRectification()` appliquent ces choix. La rectification passe par `api.ts`, puis par `ImpactEngine.rectifyPerspective()`.

Les preferences de recadrage et d'affichage sont persistees par [mobile/services/uiSettings.ts](mobile/services/uiSettings.ts) dans `AsyncStorage` :

- `manual_crop_enabled`
- `manual_crop_mode` : `frame`, `corners` ou `axes`
- `base_mode_enabled`
- `score_zones_mode` : `model` ou `detected`

### 3. Appel JS du moteur embarque

[mobile/services/api.ts](mobile/services/api.ts) est la facade utilisee par les ecrans :

- `processImage(uri)` pour le premier tir d'une serie ;
- `diffImages(beforeUri, afterUri)` pour les tirs suivants ;
- `rectifyPerspective(uri, points, outputSize)` pour le recadrage par points.

Avant chaque appel, `ensureNativeEngineReady()` verifie Android et appelle `ImpactEngine.ping()`. Le service convertit ensuite la reponse native vers les types `ProcessResult` et `DiffResult`, avec notamment :

- `impacts` ou `new_impacts` ;
- `n_impacts`, `n_new`, `n_after` ;
- `total_score`, `score_session`, `score_total` ;
- `flat_b64`, `img_width`, `img_height`.

### 4. Traitement natif Kotlin

Le traitement reel est dans [ImpactEngineModule.kt](mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEngineModule.kt). La methode publique `processImage()` appelle `runProcessFromUri()` ; le chemin general est :

1. **Lecture de l'image** : `decodeBitmapFromUri()` et `decodeBitmapFullFromUri()` lisent l'URI, appliquent l'orientation EXIF et limitent la taille de decodage.
2. **Localisation de la cible** : `detectBlackDisk()` cherche le disque noir central et estime son ellipse, son centre, ses axes et son angle.
3. **Mise a plat** : `buildFlattenMatrix()` calcule la transformation qui ramene l'ellipse a un cercle ; `flattenBitmap()` produit une image carree de `1056 x 1056` pixels.
4. **Preparation YOLO** : `runYoloInference()` redimensionne l'image dans une entree `1056 x 1056`, applique un letterbox gris, convertit en RGB float NCHW et cree le tenseur ONNX.
5. **Inference** : le modele `impact_yolo.onnx` est charge une fois par `loadYoloSession()` et execute par ONNX Runtime.
6. **Post-traitement** : les boites sous `YOLO_CONF_THRESHOLD` sont ignorees, puis `nonMaxSuppression()` supprime les doublons avec `YOLO_IOU_THRESHOLD`.
7. **Filtrage geometrique** : les detections sont ramenees dans l'image source, filtrees autour de la zone cible et converties en centres/rayons.
8. **Scoring** : `scoreFromDistanceMm()` calcule le score a partir de la distance au centre et de l'echelle `MM_PER_PX_OUT`.
9. **Serialisation** : l'image aplatie est encodee en JPEG base64 et le resultat est renvoye a JavaScript.

Le modele YOLO utilise en production est [impact_yolo.onnx](mobile/android/app/src/main/assets/impact_yolo.onnx). Le module contient aussi un chemin heatmap TFLite (`loadInterpreter()` et `runHeatmapInference()`), mais le chemin principal de `processImage()` utilise actuellement YOLO ONNX.

### 5. Resultat du premier tir

Dans `handlePhoto()` :

1. `processImage()` analyse la photo preparee.
2. `createSeries()` cree la serie avec un premier `Shot`.
3. Le resultat est place dans le cache memoire avec `setLastSeries()`.
4. L'application navigue vers `/diff`.

Le premier tir considere tous ses impacts comme nouveaux : `new_impacts` et `all_impacts` contiennent la meme liste.

### 6. Tirs suivants et comparaison

Pour un tir supplementaire, `handlePhoto()` :

1. charge l'historique avec `getHistory()` ;
2. recupere le `photo_uri` du dernier tir ;
3. prepare a nouveau la photo precedente et la nouvelle photo ;
4. appelle `diffImages(before, after)` ;
5. appelle `addShotToSeries()` ;
6. revient vers l'ecran `/diff`.

Cote natif, `diffImages()` analyse les deux images, apparie les impacts et distingue :

- les impacts deja presents ;
- les nouveaux impacts ;
- le score du tir (`score_session`) ;
- le score cumule (`score_total`).

La tolerance de matching native est `MATCH_TOL_MM = 15` mm dans `ImpactEngineModule.kt`.

## Parcours utilisateur

### Accueil

[mobile/app/index.tsx](mobile/app/index.tsx) propose :

- continuer la derniere serie ;
- demarrer une nouvelle serie ;
- ouvrir l'historique ;
- ouvrir les parametres/export.

### Camera

[mobile/app/camera.tsx](mobile/app/camera.tsx) affiche le viseur, la camera, la galerie, les modes de recadrage et l'overlay de progression. Les messages de traitement sont purement indicatifs ; le calcul est realise par le module natif.

### Serie et affichage

[mobile/app/diff.tsx](mobile/app/diff.tsx) recharge la serie, affiche les tirs dans un carousel et propose :

- vue `new` : nouveaux impacts du tir courant ;
- vue `all` : impacts cumules ;
- vue `flat` : image sans annotation ;
- zoom et deplacement de l'image ;
- affichage optionnel des zones de score ;
- selection d'un impact ;
- ajout manuel d'un impact et choix du score ;
- modification/suppression des impacts ;
- suppression du dernier tir.

Les composants responsables sont [ZoomableImage.tsx](mobile/components/ZoomableImage.tsx), [ImpactOverlay.tsx](mobile/components/ImpactOverlay.tsx), [TargetZonesOverlay.tsx](mobile/components/TargetZonesOverlay.tsx), [PlacementOverlay.tsx](mobile/components/PlacementOverlay.tsx) et [ScoreBoard.tsx](mobile/components/ScoreBoard.tsx).

Le mode de zones `detected` appelle `detectScoreRings()` et affiche les anneaux detectes via `TargetZonesOverlay`. Le mode `model` dessine les zones a partir des rayons theoriques.

### Historique et export

[mobile/app/history.tsx](mobile/app/history.tsx) lit les series, affiche la derniere image flat, les compteurs et la taille disque. [mobile/app/settings.tsx](mobile/app/settings.tsx) exporte les images flat dans une archive ZIP partageable.

## Series, comparaison et stockage

[mobile/services/storage.ts](mobile/services/storage.ts) utilise deux stockages locaux :

- `AsyncStorage` pour les metadonnees sous `@shootscore/history_v2` ;
- `expo-file-system` pour les images flat JPEG dans `documentDirectory/flats/`.

Un `SeriesRecord` contient un identifiant, une date, une liste de `Shot`, le score total et le nombre d'impacts. Chaque `Shot` contient notamment :

- `photo_uri` : URI temporaire de la photo originale ;
- `flat_file_uri` : image aplatie persistante ;
- dimensions de l'image ;
- nombre d'impacts et score ;
- `new_impacts` et `all_impacts`.

Les operations principales sont `createSeries()`, `addShotToSeries()`, `updateShotImpacts()`, `deleteLastShot()`, `deleteSeries()`, `clearHistory()` et `exportSeriesZip()`.

## Ecrans et composants

La navigation est definie dans [mobile/app/_layout.tsx](mobile/app/_layout.tsx) :

| Ecran | Fichier | Role |
|---|---|---|
| Accueil | [index.tsx](mobile/app/index.tsx) | Demarrer/reprendre une serie |
| Camera | [camera.tsx](mobile/app/camera.tsx) | Capturer, recadrer et analyser |
| Serie | [diff.tsx](mobile/app/diff.tsx) | Voir et corriger les tirs |
| Historique | [history.tsx](mobile/app/history.tsx) | Reouvrir ou supprimer une serie |
| Parametres | [settings.tsx](mobile/app/settings.tsx) | Etat du mode embarque et export |
| Diagnostic | [debug-pipeline.tsx](mobile/app/debug-pipeline.tsx) | Executer une etape native isolee |
| Resultat legacy | [result.tsx](mobile/app/result.tsx) | Ecran de resultat individuel conserve pour compatibilite |

## Pipeline Python

Le code Python sert au traitement desktop, a la creation du dataset et a l'entrainement. Il ne fait pas partie du chemin d'analyse de l'application Android.

Le pipeline complet est lance par [src/pipeline.py](src/pipeline.py) :

1. [src/localize_target.py](src/localize_target.py) localise le disque noir.
2. [src/flatten_target.py](src/flatten_target.py) corrige la perspective et produit `<stem>_flat.jpg` en `1056 x 1056`.
3. [src/detect_rings.py](src/detect_rings.py) cherche les bords d'anneaux par profil radial et genere `<stem>_rings.json`.
4. [src/detect_impacts_yolo.py](src/detect_impacts_yolo.py) execute YOLO ou ONNX, convertit les boites en impacts et calcule les scores.
5. [src/diff_shots.py](src/diff_shots.py) compare deux resultats et produit les nouveaux impacts.

Les sorties sont placees dans `outputs/<nom>/` : image flat, visualisation des anneaux, JSON de calibration, image annotee et JSON des impacts.

Les scripts heatmap et patch CNN (`src/train_heatmap_cnn.py`, `src/detect_impacts_heatmap.py`, `src/train_patch_cnn.py`) sont des pistes experimentales. Ils ne sont pas le moteur actif de l'application Android.

## Entrainement et export du modele

### Preparation des annotations

`src/label_impacts.py` permet de placer les impacts sur les images flat. `src/prepare_yolo_dataset.py` convertit les labels en dataset YOLO dans `data/yolo/`.

### Entrainement

`src/train_yolo.py` fine-tune un modele YOLO, par defaut `yolov8n.pt`, avec des images `1056 x 1056`. Le checkpoint produit est normalement :

```text
models/yolo_impacts/weights/best.pt
```

### Export Android

`src/export_onnx.py` exporte le checkpoint en ONNX et copie le fichier vers :

```text
mobile/android/app/src/main/assets/impact_yolo.onnx
```

Apres chaque nouvel export, il faut reconstruire l'application Android pour embarquer le nouveau modele.

## CNN des quatre reperes

Le mode base utilise quatre CNN independants : `top`, `left`, `bottom` et `right`. Chaque modele recoit une ROI autour de la position attendue par le viseur et renvoie la presence du chiffre `1` ainsi qu'un decalage `dx/dy`.

Le pipeline est compose de :

- [src/label_axis_markers.py](src/label_axis_markers.py) : annotation manuelle des quatre `1` ;
- [src/prepare_axis_dataset.py](src/prepare_axis_dataset.py) : extraction des quatre ROI ;
- [src/models/axis_marker_cnn.py](src/models/axis_marker_cnn.py) : architecture CNN ;
- [src/train_axis_cnn.py](src/train_axis_cnn.py) : entrainement des quatre checkpoints ;
- [src/export_axis_markers_onnx.py](src/export_axis_markers_onnx.py) : export vers Android.

### Annotation

```powershell
.\.venv\Scripts\python src/label_axis_markers.py data/raw
```

Touches : `1=haut`, `2=gauche`, `3=bas`, `4=droite`, clic pour placer, clic droit pour supprimer, `R` pour effacer, `S` pour sauver, `N/P` pour naviguer, `Q` pour quitter. Les labels sont ecrits a cote des photos sous `<photo>_axis_labels.json`.

### Preparation et entrainement

```powershell
.\.venv\Scripts\python src/prepare_axis_dataset.py data/raw --out data/axis_markers --screen-aspect 0.5625
.\.venv\Scripts\python src/train_axis_cnn.py --data data/axis_markers --epochs 80
```

Le dataset contient quatre dossiers `top`, `left`, `bottom`, `right`, chacun avec ses splits `train` et `val`. Le modele apprend la presence et le decalage du chiffre par rapport a la position attendue.

### Export et integration Android

```powershell
.\.venv\Scripts\python src/export_axis_markers_onnx.py
```

Les assets produits sont `axis_marker_top.onnx`, `axis_marker_left.onnx`, `axis_marker_bottom.onnx` et `axis_marker_right.onnx` dans `mobile/android/app/src/main/assets/`. [ImpactEngineModule.kt](mobile/android/app/src/main/java/com/rexmi/shootscore/impact/ImpactEngineModule.kt) les charge via ONNX Runtime ; en leur absence, il conserve temporairement la detection classique.

## Installation et commandes

### Environnement Python

```powershell
uv sync
```

### Pipeline desktop

```powershell
.\.venv\Scripts\python src/pipeline.py data/raw/photo.jpg --show
.\.venv\Scripts\python src/pipeline.py data/raw --out outputs --show
.\.venv\Scripts\python src/diff_shots.py avant.jpg apres.jpg --tol 8 --show
```

### Entrainement et export

```powershell
.\.venv\Scripts\python src/prepare_yolo_dataset.py outputs/flatten --out data/yolo --val-ratio 0.2
.\.venv\Scripts\python src/train_yolo.py --data data/yolo/dataset.yaml --epochs 200 --batch 8 --imgsz 1056
.\.venv\Scripts\python src/export_onnx.py
```

### Developpement mobile

```powershell
Push-Location mobile
npm install
npx expo start
Pop-Location
```

Expo Go peut servir a travailler sur l'interface, mais il ne charge pas le module Kotlin `ImpactEngine`.

### Build Android avec le moteur natif

```powershell
Push-Location mobile\android
.\gradlew.bat app:compileDebugKotlin
.\gradlew.bat app:installRelease
Pop-Location
adb devices
adb shell monkey -p com.rexmi.shootscore -c android.intent.category.LAUNCHER 1
```

Une build Android personnalisee est obligatoire pour tester l'inference embarquee.

## Diagnostic

[mobile/app/debug-pipeline.tsx](mobile/app/debug-pipeline.tsx) appelle directement les methodes de diagnostic du module natif :

- `debugDetectDisk` : detection du disque noir ;
- `debugFlatten` : correction de perspective et image flat ;
- `debugDetectImpactsFlat` : detection sur une image deja flat ;
- `debugDetectImpactsFull` : pipeline complet sur une image source.

Les parametres sont `step=disk|flatten|impacts-flat|impacts-full` et `uri=file://...`. Les resultats sont affiches dans l'ecran et dans les logs Metro/ADB.

## Limites actuelles

- L'inference embarquee est disponible sur Android uniquement.
- Le modele charge par Android est l'asset ONNX ; modifier un checkpoint Python ne modifie pas l'application tant que l'export et la reconstruction Android ne sont pas faits.
- Il n'existe pas encore de suite de tests end-to-end couvrant camera, inference native, comparaison et persistance.
- `result.tsx` est un ecran ancien ; le parcours courant passe par `diff.tsx`.
