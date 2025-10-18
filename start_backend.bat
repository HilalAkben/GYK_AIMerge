@echo off
echo Backend API baslatiliyor...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
cd api
python main.py
pause

