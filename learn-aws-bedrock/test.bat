@echo off
rem Run the offline test suite (no AWS account needed).
rem Optional: set PYTHON_EXE to the full path of the Python you want to use.
setlocal
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
cd /d "%~dp0"
"%PYTHON_EXE%" -m pytest tests -q %*
exit /b %ERRORLEVEL%
