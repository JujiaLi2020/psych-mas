$ErrorActionPreference = "Stop"
$compose = Join-Path $PSScriptRoot "docker-compose.release.yml"
$envFile = Join-Path $PSScriptRoot ".env"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is not installed. Run PsyMAS Setup first."
}

& docker desktop start *> $null
$deadline = (Get-Date).AddMinutes(4)
do {
    & docker info *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 4
} while ((Get-Date) -lt $deadline)

& docker compose --env-file $envFile -f $compose up -d
if ($LASTEXITCODE -ne 0) { throw "PsyMAS could not be started." }
Start-Process "http://localhost:8501"
