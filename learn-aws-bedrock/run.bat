@echo off
rem Run one lesson:   run.bat b01_converse_loop --offline
rem Optional: set PYTHON_EXE to the full path of the Python you want to use.
setlocal
if "%~1"=="" (
  echo Usage: run.bat ^<chapter-folder^> [arguments]
  echo Example: run.bat b01_converse_loop --offline
  exit /b 1
)
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
set "CHAPTER=%~1"
shift
"%PYTHON_EXE%" "%~dp0%CHAPTER%\code.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%
