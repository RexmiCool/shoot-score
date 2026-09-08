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

---

# Boucle globale de collecte, validation et entraînement

## Objectif

Mettre en place une boucle continue de données pour construire progressivement les datasets de repères puis d’impacts, avec validation humaine à chaque étape, sans mélange entre données brutes, données auto-labelisées et données validées.

La logique générale est la suivante :

- incoming = nouvelles photos arrivées dans le flux
- pending = images en attente de validation
- accepted = images validées et utilisables pour l’entraînement
- rejected = images rejetées

Le principe est simple :

- si un modèle existe, on auto-labelise les images
- si aucun modèle n’existe, on copie les images dans pending pour les labeliser manuellement
- ensuite on valide manuellement le résultat et on envoie vers accepted ou rejected
- puis seulement les images de accepted sont utilisées pour entraîner le modèle

Règle fondamentale du flux incrémental :

- on ne traite jamais tout le stock complet à chaque fois
- on ne traite que les nouvelles photos à chaque boucle
- les images déjà validées et déjà traitées ne sont pas retravaillées
- le modèle est ré-entraîné sur le cumul des images validées jusqu’à ce moment

Exemple :

- Jour 1 : une photo p1 arrive dans incoming, elle est gardée puis validée dans axis, puis dans impact, puis le modèle est entraîné sur p1
- Jour 2 : une nouvelle photo p2 arrive dans incoming, elle est gardée puis elle passe seulement dans la nouvelle boucle de traitement
- p1 n’est pas retravaillé car il est déjà labelisé et validé
- le modèle est ensuite ré-entraîné sur {p1, p2}
- puis on continue avec p3, p4, etc.

Cela permet d’avoir une boucle continue, simple et propre, sans refouler les anciennes données ni recalculer inutilement tout le dataset.

---

## Structure cible des dossiers

```text
original_photos/
  incoming/
  accepted/
  rejected/

axis_labelled/
  pending/
  accepted/
  rejected/

homographied/
  pending/
  accepted/
  rejected/

impact_labelled/
  pending/
  accepted/
  rejected/
```

Les dossiers sont organisés par étape du pipeline, et chaque étape suit la même logique de tri :

- pending : fichiers à inspecter ou résultats automatiques à valider
- accepted : fichiers validés et prêts pour la suite
- rejected : fichiers rejetés

---

## Flux complet du pipeline

### 1) Entrée des nouvelles images

On part du dossier racine :

```text
original_photos/
  incoming/
  accepted/
  rejected/
```

Processus :

1. Les nouvelles photos arrivent dans incoming.
2. Une personne les inspecte.
3. Elle décide soit de les garder dans accepted, soit de les mettre dans rejected.

À ce stade, seul le flux accepted est utilisé pour la suite.

---

### 2) Détection des 4 repères "axis"

On prend uniquement les nouvelles images qui viennent d’être validées dans original_photos/accepted.

Cas A : un modèle de détection des 4 repères existe déjà

- on lance l’inférence automatique uniquement sur les nouvelles images
- les résultats sont placés dans axis_labelled/pending
- puis validation manuelle : accepted ou rejected

Cas B : aucun modèle de repères n’existe encore

- on copie les nouvelles images dans axis_labelled/pending
- puis on les labelise manuellement
- ensuite on valide vers axis_labelled/accepted ou axis_labelled/rejected

Important :

- les images déjà labelisées et validées ne sont pas repassées dans le traitement
- le modèle est entraîné sur le cumul des images validées jusqu’à présent
- donc à chaque nouvel ajout, on ne traite que la nouveauté

Une fois les nouvelles images dans axis_labelled/accepted, elles sont prêtes pour la suite.

---

### 3) Homographie

À partir des images validées dans axis_labelled/accepted :

- on applique la transformation perspective / homographie
- les images résultantes sont envoyées dans homographied/pending
- puis validation humaine :
  - accepted = bonne image homogénéisée
  - rejected = image rejetée

Le résultat final de cette étape est :

```text
homographied/accepted
```

---

### 4) Détection des impacts

À partir uniquement des nouvelles images qui viennent d’être validées dans homographied/accepted :

Cas A : un modèle YOLO impacts existe déjà

- on lance l’inférence automatique seulement sur les nouvelles images
- les résultats vont dans impact_labelled/pending
- ensuite validation manuelle : accepted ou rejected

Cas B : aucun modèle impacts n’existe encore

- on copie les nouvelles images dans impact_labelled/pending
- on les labelise manuellement
- puis validation vers impact_labelled/accepted ou impact_labelled/rejected

Important :

- on ne re-labelise pas l’historique déjà validé
- on traite uniquement le nouveau lot en arrivée
- le modèle est ensuite ré-entraîné sur la somme cumulée des images validées jusqu’à ce moment

Une fois validées, les nouvelles images dans impact_labelled/accepted servent à construire le dataset d’entraînement.

---

### 5) Entraînement du modèle impacts

Quand il y a suffisamment d’images dans impact_labelled/accepted :

1. on prépare le dataset YOLO
2. on entraîne le modèle YOLO impacts
3. on vérifie la qualité du modèle
4. on exporte le modèle vers ONNX
5. on intègre le modèle dans l’application Android

---

## Règles de validation humaine

Chaque étape de validation suit la même convention :

- pending = résultat automatique ou images à vérifier
- accepted = images validées
- rejected = images rejetées

La personne ne valide jamais directement un dataset plein :

- elle inspecte le dossier pending
- elle confirme ou rejette chaque image
- elle décide de son statut final

Cette règle évite de mélanger des images “non vérifiées” avec des images “trainables”.

---

## Plan d’action étape par étape

### Étape 1 : mettre en place la structure de dossiers

Créer les dossiers suivants :

```text
original_photos/
  incoming/
  accepted/
  rejected/

axis_labelled/
  pending/
  accepted/
  rejected/

homographied/
  pending/
  accepted/
  rejected/

impact_labelled/
  pending/
  accepted/
  rejected/
```

Objectif : préparer la boucle de données avant même le premier entraînement.

---

### Étape 2 : initialiser le tri des nouvelles photos

- mettre les nouvelles photos dans original_photos/incoming
- faire le tri en accepted ou rejected
- ne garder que les images validées dans original_photos/accepted
- ne pas réintroduire dans le pipeline les images déjà traitées

Objectif : créer un stock de données propres et exploitable, tout en ne traitant que la nouveauté à chaque tour.

---

### Étape 3 : créer le dataset axis

On ne traite que les nouvelles images validées dans original_photos/accepted.

Si le modèle axis existe déjà :

- lancer l’inférence automatique uniquement sur les nouvelles images
- envoyer dans axis_labelled/pending
- valider vers accepted ou rejected

Sinon :

- copier les nouvelles images dans axis_labelled/pending
- labeliser manuellement
- valider vers axis_labelled/accepted ou rejected

Ensuite :

- créer le dataset YOLO axis sur le cumul des images validées
- entraîner le modèle
- exporter en ONNX
- intégrer le modèle dans l’application

Le point important est que p1 ne redevient pas une photo à relabeliser le jour 2 ; on traite uniquement p2.

---

### Étape 4 : générer les images homogénéisées

- prendre axis_labelled/accepted
- appliquer la homographie
- enregistrer dans homographied/pending
- valider vers homographied/accepted ou rejected

Objectif : obtenir un set d’images plates, homogénéisées et prêtes pour la détection d’impacts.

---

### Étape 5 : créer le dataset impacts

On ne traite que les nouvelles images validées dans homographied/accepted.

Si le modèle impact existe déjà :

- run auto-detection uniquement sur les nouvelles images
- copier dans impact_labelled/pending
- valider vers accepted ou rejected

Sinon :

- copier les nouvelles images dans impact_labelled/pending
- labeliser manuellement
- valider vers impact_labelled/accepted ou rejected

L’historique déjà validé est conservé et n’est pas recalculé.

---

### Étape 6 : entraîner le modèle impacts

- préparer le dataset YOLO impacts
- entraîner le modèle
- valider la performance sur un lot de test
- exporter le modèle ONNX
- intégrer le modèle dans l’application mobile

---

### Étape 7 : boucler la collecte

Une fois le pipeline fonctionnel :

1. les nouvelles photos arrivent dans incoming
2. elles sont triées dans accepted ou rejected
3. elles passent uniquement dans le pipeline axis
4. puis dans l’étape homographie
5. puis dans le pipeline impact
6. puis elles sont ajoutées au cumul des données validées
7. le modèle est ré-entraîné sur le cumul total actuel

Le modèle s’améliore au fil du temps sans repartir de zéro et sans re-traiter les anciennes photos déjà validées.

Exemple concret :

- Jour 1 : p1 arrive, est validée et entraînée
- Jour 2 : p2 arrive, est validée puis ré-entraînée avec {p1, p2}
- Jour 3 : p3 arrive, est validée puis ré-entraînée avec {p1, p2, p3}

C’est bien cette logique de traitement incrémental que nous voulons.

---

## Objectif final

La bonne boucle est donc la suivante :

```text
incoming
  -> accepted / rejected
  -> axis_labelled/pending
  -> validation
  -> axis_labelled/accepted / rejected
  -> homographied/pending
  -> validation
  -> homographied/accepted / rejected
  -> impact_labelled/pending
  -> validation
  -> impact_labelled/accepted / rejected
  -> entraînement YOLO
  -> export ONNX
  -> application
```

Cette logique donne un système robuste, traçable et évolutif, où chaque image est soit validée, soit rejetée, jamais laissée dans un état ambigu.

---

## Recommandation d’implémentation

Pour l’implémentation technique, il faut prévoir :

- un script de tri initial pour incoming
- un script de copie vers pending selon l’état du modèle
- un script de validation manuelle accepted/rejected
- un script de préparation dataset YOLO
- un script de training
- un script d’export ONNX
- un script global de pipeline complet

L’idée est d’avoir un workflow automatisé avec une validation humaine explicite à chaque étape, plutôt qu’un pipeline opaque ou entièrement automatique.

---

## Résumé court

Le pipeline que nous voulons est une boucle de données progressive :

- nouvelles photos dans incoming
- tri manuel vers accepted/rejected
- auto-labelisation si modèle existant, sinon copie en pending pour validation
- validation manuelle vers accepted/rejected
- entraînement des modèles après chaque lot validé
- export ONNX et mise à jour de l’application

C’est la bonne base pour faire évoluer le projet sans perdre la qualité des labels ni la traçabilité des données.
