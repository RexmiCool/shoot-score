@echo off
title ShootScore — Serveur API
cd /d "%~dp0"
echo.
echo  ╔══════════════════════════════════════╗
echo  ║       ShootScore — Serveur API       ║
echo  ╚══════════════════════════════════════╝
echo.

:: Affiche l'IP locale pour faciliter la config de l'app
echo  Adresses IP disponibles :
ipconfig | findstr /i "IPv4"
echo.

:: Démarre le serveur
echo  Démarrage sur http://0.0.0.0:8000 ...
echo  (Ctrl+C pour arrêter)
echo.
.venv\Scripts\uvicorn src.api:app --host 0.0.0.0 --port 8000

pause
