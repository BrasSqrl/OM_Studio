@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :install

python -m venv .venv
if errorlevel 1 goto :fail

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -e .[dev]
if errorlevel 1 goto :fail

echo.
echo OM Studio environment is ready.
exit /b 0

:fail
echo.
echo OM Studio setup failed.
pause
exit /b 1
