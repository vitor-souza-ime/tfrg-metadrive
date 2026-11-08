@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
%PY% -m pytest -q
if errorlevel 1 goto FAILED
if not exist results\PRECHECK_PASSED.json (
    echo Mandatory traffic precheck has not been executed.
    echo Running preflight now...
    %PY% main.py --profile preflight --skip-visual
    if errorlevel 1 goto FAILED
)
%PY% main.py --profile full
if errorlevel 1 goto FAILED
echo.
echo Full experiment finished. Send the generated _SEND.zip.
pause
exit /b 0
:FAILED
echo.
echo Experiment stopped. Preserve the current results folder.
pause
exit /b 1
