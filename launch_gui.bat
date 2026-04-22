@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call "%~dp0setup_gui.bat"
  if errorlevel 1 goto :fail
)

".venv\Scripts\python.exe" -m quant_studio_monitoring.gui_launcher
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo OM Studio failed to launch.
pause
exit /b 1
