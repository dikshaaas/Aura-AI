@echo off
echo ==========================================
echo AURA.AI: Modular Accent Detector setup
echo ==========================================

echo.
echo 1. Creating python virtual environment in venv/...
py -m venv venv
if %errorlevel% neq 0 (
    echo [warning] 'py' launcher failed, falling back to 'python' executable...
    python -m venv venv
)

if not exist venv\Scripts\activate.bat (
    echo [error] Failed to initialize virtual environment.
    exit /b 1
)

echo.
echo 2. Installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 3. Cleaning up old redundant root files (migrated to backend/ and ml/)...
if exist app.py del app.py
if exist preprocess.py del preprocess.py
if exist train.py del train.py
if exist passages.json del passages.json
if exist analyze.py del analyze.py

echo.
echo 4. Generating acoustic features dataset from recordings/ mapping...
python ml/feature_engineering/extract_features.py

echo.
echo 5. Training SVM accent classifier model...
python ml/training/train_model.py

echo.
echo 6. Running classifier evaluations...
python ml/training/evaluate_model.py

echo.
echo 7. Launching Flask backend REST service...
echo Open http://localhost:5000 in your browser to run accent recognitions!
python backend/app.py
