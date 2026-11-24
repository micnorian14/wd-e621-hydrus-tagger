@echo off
rem Tags the files listed in your input file (see config.yaml), or in the
rem text file you drop onto this bat. Settings come from config.yaml.
rem run.bat and dry-run.bat are the same except dry-run.bat adds --dry-run,
rem which does everything except write tags to Hydrus.
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set "PY=python"
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if "%PY%"=="python" echo Note: no venv folder next to this file, so using python from PATH.
"%PY%" -m wd-hydrus-tagger %*
pause
