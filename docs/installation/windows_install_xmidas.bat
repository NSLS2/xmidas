@echo off
setlocal

REM --- Clone repo if not already present ---
IF NOT EXIST xmidas (
    echo Cloning XMIDAS from GitHub...
    git clone https://github.com/pattammattel/xmidas.git
)

REM --- Change to xmidas directory ---
cd xmidas

REM --- Confirm location ---
IF NOT EXIST requirements.txt (
    echo You are not in the xmidas folder. Exiting.
    pause
    exit /b
)

REM --- Create Conda env if it doesn't exist ---
echo Creating Conda environment xmidas-env...
conda info --envs | findstr "xmidas-env" >nul
IF %ERRORLEVEL% NEQ 0 (
    conda create -n xmidas-env python=3.12 -y
)

REM --- Activate the environment ---
call conda activate xmidas-env

REM --- Install dependencies ---
echo Installing Python dependencies...
pip install -r requirements.txt

REM --- Launch XMIDAS ---
echo Starting XMIDAS...
python -m xmidas.main

pause
