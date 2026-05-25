# old

Ce dossier contient les fichiers de l'ancienne architecture, sortis de la racine du projet pour garder une base plus propre autour de la version mobile embarquee.

## Contenu

- `old/server/`: ancienne couche serveur (FastAPI, Docker, scripts de lancement)
- `old/experiments/src/`: scripts Python legacy / experimentaux
- `old/mobile/components/`: anciens composants mobile lies au mode client-serveur

## Note

Ces fichiers sont archives, pas supprimes. Ils restent disponibles pour reference ou restauration.

## Fichiers deplaces

- `Dockerfile` -> `old/server/Dockerfile`
- `docker-compose.yml` -> `old/server/docker-compose.yml`
- `docker-compose.rpi.yml` -> `old/server/docker-compose.rpi.yml`
- `start-server.bat` -> `old/server/start-server.bat`
- `shoot-score-api-rpi.tar` -> `old/server/shoot-score-api-rpi.tar`
- `src/api.py` -> `old/server/api.py`
- `src/detect_cv.py` -> `old/experiments/src/detect_cv.py`
- `src/detect_impacts.py` -> `old/experiments/src/detect_impacts.py`
- `src/tune_impacts.py` -> `old/experiments/src/tune_impacts.py`
- `src/crop_only.py` -> `old/experiments/src/crop_only.py`
- `src/debug_disk.py` -> `old/experiments/src/debug_disk.py`
- `src/diff_impacts.py` -> `old/experiments/src/diff_impacts.py`
- `src/find_black_disk.py` -> `old/experiments/src/find_black_disk.py`
- `src/target_utils.py` -> `old/experiments/src/target_utils.py`
- `mobile/components/ServerConfig.tsx` -> `old/mobile/components/ServerConfig.tsx`
