Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "LawMate AI - Legal Consultation Backend (Windows)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Change to the directory of the script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($scriptDir) { Set-Location $scriptDir }

# Check for python
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH! Please install Python."
    Exit
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
# Using location-relative path for activation script
$ActivateScript = Join-Path (Get-Location) "venv\Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    . $ActivateScript
} else {
    Write-Host "Warning: Could not find $ActivateScript. Trying standard activate.bat or running without venv." -ForegroundColor Red
}

# Upgrade pip and install dependencies
Write-Host "Installing/Updating dependencies (this may take a few minutes)..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Starting LawMate AI Backend Server..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Server will start on: http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Green
Write-Host ""

python app.py
