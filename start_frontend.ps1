Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "LawMate AI - ChatLaw Frontend (Windows)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Change to the directory of the script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($scriptDir) { Set-Location $scriptDir }

# Check for node/npm
if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
    Write-Error "NodeJS/npm is not installed or not in PATH! Please install NodeJS."
    Exit
}

# Install dependencies if node_modules doesn't exist
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
} else {
    Write-Host "Frontend dependencies (node_modules) already present." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Starting Vite Development Server..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Server will start on: http://localhost:5173" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Green
Write-Host ""

npm run dev
