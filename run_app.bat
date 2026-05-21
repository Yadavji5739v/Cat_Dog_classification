@echo off
echo ============================================================
echo        PawPredict AI  -  Cat vs Dog Classifier
echo ============================================================
echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
echo.
echo [2/3] Checking for model file...
if exist "models\cats_dogs_cnn_model.keras" (
    echo  Found: models\cats_dogs_cnn_model.keras
) else if exist "models\cats_dogs_cnn_model.h5" (
    echo  Found: models\cats_dogs_cnn_model.h5
) else (
    echo  WARNING: No model file found!
    echo  Please copy your trained model into the 'models' folder:
    echo    - models\cats_dogs_cnn_model.keras   OR
    echo    - models\cats_dogs_cnn_model.h5
    echo.
    echo  Creating 'models' folder for you...
    mkdir models 2>nul
)
echo.
echo [3/3] Launching Streamlit app...
echo  Open your browser at: http://localhost:8501
echo.
streamlit run app.py
pause
