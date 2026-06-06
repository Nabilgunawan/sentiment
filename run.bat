@echo off
echo.
echo ===== Sentiment Analyzer =====
echo.
cd /d "C:\Users\Nabill\AppData\Local\Temp\opencode\sentiment-app"
echo Folder: %CD%
echo.
echo 1. Memeriksa Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak ditemukan!
    echo Silakan install Python dari: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo    Python OK.
echo.
echo 2. Menginstall dependencies...
python -m pip install --upgrade pip wheel
python -m pip install --only-binary :all: -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Gagal install dependencies.
    pause
    exit /b 1
)
echo    Dependencies OK.
echo.
echo 3. Menjalankan server...
echo.
echo Buka browser ke: http://localhost:5000
echo Tekan Ctrl+C untuk menghentikan server.
echo.
python app.py
pause
