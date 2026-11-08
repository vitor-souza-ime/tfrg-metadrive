@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
echo Checking Python version...
python -c "import sys; assert (3,10) <= sys.version_info[:2] < (3,12), 'Python 3.10 or 3.11 required for this package'; print(sys.version)"
if errorlevel 1 goto FAILED
if not exist .venv\Scripts\python.exe (
    python -m venv .venv
    if errorlevel 1 goto FAILED
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto FAILED
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto FAILED
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto FAILED
.venv\Scripts\python.exe -m pytest -q
if errorlevel 1 goto FAILED
echo.
echo Installation and unit tests completed.
pause
exit /b 0
:FAILED
echo.
echo Installation failed. Check the terminal output before proceeding.
pause
exit /b 1
