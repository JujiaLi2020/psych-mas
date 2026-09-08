[CmdletBinding()]
param(
    [switch]$SkipDockerInstall,
    [switch]$SkipLlmSetup
)

$ErrorActionPreference = "Stop"
$ComposeFile = Join-Path $PSScriptRoot "docker-compose.release.yml"
$EnvFile = Join-Path $PSScriptRoot ".env"
$AppDataRoot = Join-Path $env:LOCALAPPDATA "PsyMAS"
$RunDataPath = Join-Path $AppDataRoot "data\output"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Test-DockerReady {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
    & docker info *> $null
    return $LASTEXITCODE -eq 0
}

function Start-DockerDesktop {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        & docker desktop start *> $null
        if ($LASTEXITCODE -eq 0) { return }
    }

    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Docker\Docker\Docker Desktop.exe",
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    )
    $desktop = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($desktop) { Start-Process $desktop }
}

Write-Host "PsyMAS v0.7.5 Setup" -ForegroundColor White
Write-Host "This installer keeps assessment and review data in $RunDataPath."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    if ($SkipDockerInstall) {
        throw "Docker Desktop is required but was not found."
    }

    Write-Step "Docker Desktop is required"
    $answer = Read-Host "Install Docker Desktop using Windows Package Manager now? [Y/n]"
    if ($answer -and $answer -notmatch '^[Yy]') {
        Start-Process "https://docs.docker.com/desktop/setup/install/windows-install/"
        throw "Install and start Docker Desktop, then run PsyMAS Setup again."
    }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Start-Process "https://www.docker.com/products/docker-desktop/"
        throw "Windows Package Manager is unavailable. Complete the Docker Desktop installer, then run PsyMAS Setup again."
    }
    & winget install --exact --id Docker.DockerDesktop --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop installation did not complete successfully." }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

Write-Step "Starting Docker Desktop"
Start-DockerDesktop
$deadline = (Get-Date).AddMinutes(6)
while ((Get-Date) -lt $deadline -and -not (Test-DockerReady)) {
    Start-Sleep -Seconds 5
}
if (-not (Test-DockerReady)) {
    throw "Docker did not become ready. Open Docker Desktop, complete its first-run setup, and run PsyMAS Setup again. A restart may be required after enabling WSL 2."
}

New-Item -ItemType Directory -Force -Path $RunDataPath | Out-Null
$dockerDataPath = $RunDataPath.Replace('\', '/')
$settings = [ordered]@{
    PSYMAS_IMAGE_TAG = "0.7.5"
    PSYMAS_DATA_DIR = $dockerDataPath
    OPENROUTER_API_KEY = ""
}

if (Test-Path $EnvFile) {
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        if ($line -match '^([^#=]+)=(.*)$') { $settings[$matches[1].Trim()] = $matches[2] }
    }
}
# An upgrade may reuse the existing .env, but the application image must match
# the installer version. User-managed data paths and LLM settings remain intact.
$settings.PSYMAS_IMAGE_TAG = "0.7.5"

if (-not $SkipLlmSetup) {
    Write-Step "Configure optional AI-assisted reporting"
    Write-Host "1. OpenRouter API (recommended)"
    Write-Host "2. No AI"
    Write-Host "3. Local Ollama (advanced)"
    $choice = Read-Host "Choose 1, 2, or 3 [1]"
    if (-not $choice) { $choice = "1" }

    switch ($choice) {
        "1" {
            $key = Read-Host "Enter your OpenRouter API key, or press Enter to configure it later"
            $settings.OPENROUTER_API_KEY = $key.Trim()
            $settings.Remove("OLLAMA_CHAT_URL")
        }
        "3" {
            $settings.OPENROUTER_API_KEY = ""
            $settings.OLLAMA_CHAT_URL = "http://host.docker.internal:11434/api/chat"
            Write-Host "Install Ollama separately, run 'ollama pull llama3.1:8b', and keep Ollama running."
            Start-Process "https://ollama.com/download/windows"
        }
        default {
            $settings.OPENROUTER_API_KEY = ""
            $settings.Remove("OLLAMA_CHAT_URL")
        }
    }
}

$envLines = @($settings.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" })
[System.IO.File]::WriteAllLines($EnvFile, $envLines, (New-Object System.Text.UTF8Encoding($false)))

Write-Step "Downloading the versioned PsyMAS image"
& docker compose -p psymas-desktop --env-file $EnvFile -f $ComposeFile pull
if ($LASTEXITCODE -ne 0) { throw "Could not download the PsyMAS container image." }

Write-Step "Starting PsyMAS"
& docker compose -p psymas-desktop --env-file $EnvFile -f $ComposeFile up -d
if ($LASTEXITCODE -ne 0) { throw "PsyMAS services could not be started." }

$healthDeadline = (Get-Date).AddMinutes(3)
$healthy = $false
while ((Get-Date) -lt $healthDeadline) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 5
        if ($response.StatusCode -eq 200) { $healthy = $true; break }
    } catch { Start-Sleep -Seconds 4 }
}
if (-not $healthy) {
    Write-Warning "PsyMAS is still starting. Check 'docker compose -f `"$ComposeFile`" logs backend'."
}

Write-Host "`nPsyMAS is ready at http://localhost:8501" -ForegroundColor Green
Start-Process "http://localhost:8501"
