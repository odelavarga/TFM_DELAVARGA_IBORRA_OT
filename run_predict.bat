@echo off
REM Helper script per executar predict.py amb l'entorn virtual
REM Ús: run_predict.bat [arguments]
REM Exemples:
REM   run_predict.bat --input examples/habitatge_exemple.json
REM   run_predict.bat --metros 90 --habitaciones 3 --aseos 2 --terraza 1 --garaje 1 --provincia Barcelona
REM   run_predict.bat --batch examples/habitatges_batch.csv

cd /d "%~dp0"
env\Scripts\python.exe predict.py %*
