Write-Host "Starting Splitwise Receipt Scanner Demo..." -ForegroundColor Green
Write-Host ""

Write-Host "1. Starting FastAPI server in background..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-Command", "python main.py" -WindowStyle Normal

Write-Host "2. Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "3. Starting Streamlit app..." -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Splitwise Receipt Scanner is now running!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   FastAPI Server: http://localhost:8000" -ForegroundColor White
Write-Host "   Streamlit App: http://localhost:8501" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py