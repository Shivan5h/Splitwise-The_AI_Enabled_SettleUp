@echo off
echo Starting Splitwise Receipt Scanner Demo...
echo.

echo 1. Starting FastAPI server in background...
start "FastAPI Server" cmd /k "python main.py"

echo 2. Waiting for server to start...
timeout /t 5 /nobreak >nul

echo 3. Starting Streamlit app...
echo.
echo ================================================
echo   Splitwise Receipt Scanner is now running!
echo ================================================
echo   FastAPI Server: http://localhost:8000
echo   Streamlit App: http://localhost:8501
echo ================================================
echo.

streamlit run app.py