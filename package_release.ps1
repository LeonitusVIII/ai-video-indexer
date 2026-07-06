# Build a clean zip of AI Video Indexer for sharing.
# Usage:  .\package_release.ps1
# Output: dist\VideoIndexer.zip

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$OutDir = Join-Path $Root "dist"
$Staging = Join-Path $OutDir "VideoIndexer"
$ZipPath = Join-Path $OutDir "VideoIndexer.zip"

$IncludeFiles = @(
    "app.py",
    "app_helpers.py",
    "app_jobs.py",
    "app_wizard.py",
    "app_help.py",
    "app_db.py",
    "pipeline_estimate.py",
    "search_engine.py",
    "notifications.py",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "config.example.json",
    "setup.bat",
    "start.bat",
    ".gitignore"
)

$IncludeDirs = @(
    "scripts"
)

Write-Host "Building release package..." -ForegroundColor Cyan

if (Test-Path $Staging) {
    Remove-Item $Staging -Recurse -Force
}
New-Item -ItemType Directory -Path $Staging -Force | Out-Null

foreach ($file in $IncludeFiles) {
    $src = Join-Path $Root $file
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $Staging $file) -Force
    }
}

foreach ($dir in $IncludeDirs) {
    $src = Join-Path $Root $dir
    if (Test-Path $src) {
        $dest = Join-Path $Staging $dir
        Copy-Item $src $dest -Recurse -Force
        Get-ChildItem $dest -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem $dest -Recurse -File -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

# Fresh config and empty runtime folders for the recipient
Copy-Item (Join-Path $Root "config.example.json") (Join-Path $Staging "config.json") -Force

foreach ($sub in @("data", "jobs", "logs")) {
    $path = Join-Path $Staging $sub
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    New-Item -ItemType File -Path (Join-Path $path ".gitkeep") -Force | Out-Null
}

Copy-Item (Join-Path $Root "package_release.ps1") (Join-Path $Staging "package_release.ps1") -Force

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $ZipPath -CompressionLevel Optimal

$sizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
Write-Host ""
Write-Host "Created: $ZipPath ($sizeMb MB)" -ForegroundColor Green
Write-Host "Send this zip to your friend. They run setup.bat then start.bat." -ForegroundColor Green
