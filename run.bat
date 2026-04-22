@echo off
setlocal enabledelayedexpansion

set USE_STREAMLIT=false
set SKIP_EXTRACT=false
set SKIP_TRAIN=false

for %%A in (%*) do (
    if "%%A"=="--streamlit"    set USE_STREAMLIT=true
    if "%%A"=="--skip-extract" set SKIP_EXTRACT=true
    if "%%A"=="--skip-train"   set SKIP_TRAIN=true
)

echo.
echo   =====================================================
echo    NARUTO Hand Sign Detector - Jutsu Activation System
echo   =====================================================
echo.

:: ── Dataset ────────────────────────────────────────────────────
echo.
echo [INFO]  Checking dataset...
set MISSING=0
for %%C in (bird boar dog dragon hare horse monkey ox ram rat snake tiger zero) do (
    if not exist "data\train\%%C" ( echo [WARN]  Missing: data\train\%%C & set /a MISSING+=1 )
    if not exist "data\test\%%C"  ( echo [WARN]  Missing: data\test\%%C  & set /a MISSING+=1 )
)
if !MISSING! GTR 0 (
    echo [ERROR] Dataset incomplete.
    pause & exit /b 1
)
echo [OK]    All 13 class folders found

:: ── Python ─────────────────────────────────────────────────────
echo.
echo [INFO]  Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org ^(tick "Add to PATH"^)
    pause & exit /b 1
)
for /f "tokens=*" %%V in ('python --version 2^>^&1') do echo [OK]    Using %%V

:: ── Write dependency helper script ────────────────────────────
echo.
echo [INFO]  Checking dependencies...

echo import importlib.util, subprocess, sys > _check_deps.py
echo pkgs   = ['mediapipe','opencv-python','numpy','pandas','scikit-learn','xgboost','catboost','joblib','tqdm','streamlit'] >> _check_deps.py
echo checks = ['mediapipe','cv2','numpy','pandas','sklearn','xgboost','catboost','joblib','tqdm','streamlit'] >> _check_deps.py
echo missing = [p for p,c in zip(pkgs,checks) if importlib.util.find_spec(c) is None] >> _check_deps.py
echo if missing: >> _check_deps.py
echo     print('[INFO]  Installing: ' + ' '.join(missing)) >> _check_deps.py
echo     subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing + ['-q']) >> _check_deps.py
echo     print('[OK]    Dependencies installed') >> _check_deps.py
echo else: >> _check_deps.py
echo     print('[OK]    All dependencies already installed') >> _check_deps.py

python _check_deps.py
if errorlevel 1 (
    echo [ERROR] Dependency install failed. Try running as Administrator.
    del _check_deps.py
    pause & exit /b 1
)
del _check_deps.py

:: ── Step 1 - Extract landmarks ─────────────────────────────────
echo.
if "%SKIP_EXTRACT%"=="true" (
    echo [WARN]  Skipping extraction ^(--skip-extract^)
    if not exist "data\landmarks_train.csv" ( echo [ERROR] landmarks_train.csv not found. & pause & exit /b 1 )
    if not exist "data\landmarks_test.csv"  ( echo [ERROR] landmarks_test.csv not found.  & pause & exit /b 1 )
    echo [OK]    Found existing CSVs
) else (
    echo ====  Step 1 - Extracting landmarks  ====
    echo [INFO]  Running MediaPipe on all images... ^(2-5 min^)
    echo.
    python step1_extract_landmarks.py
    if errorlevel 1 ( echo [ERROR] Extraction failed. & pause & exit /b 1 )
    echo [OK]    Landmark CSVs saved
)

:: ── Step 2 - Train ─────────────────────────────────────────────
echo.
if "%SKIP_TRAIN%"=="true" (
    echo [WARN]  Skipping training ^(--skip-train^)
    if not exist "models\ensemble.pkl" ( echo [ERROR] ensemble.pkl not found. & pause & exit /b 1 )
    echo [OK]    Found existing model
) else (
    echo ====  Step 2 - Training model  ====
    echo [INFO]  RF + XGBoost + CatBoost soft voting... ^(5-15 min^)
    echo.
    python step2_train_model.py
    if errorlevel 1 ( echo [ERROR] Training failed. & pause & exit /b 1 )
    echo [OK]    Model saved to models\ensemble.pkl
)

:: ── Step 3 - Launch ────────────────────────────────────────────
echo.
echo ====  Step 3 - Launching app  ====
if "%USE_STREAMLIT%"=="true" (
    echo [INFO]  Streamlit - opening http://localhost:8501
    echo [INFO]  Press Ctrl+C to stop
    echo.
    start "" "http://localhost:8501"
    streamlit run app_streamlit.py -- --model models\ensemble.pkl
) else (
    echo [INFO]  OpenCV app  - Q/ESC=quit  C=clear  S=screenshot
    echo.
    python step3_realtime.py --model models\ensemble.pkl
)

echo.
echo [OK]    Done.
pause
