$ErrorActionPreference = "Stop"
$compose = Join-Path $PSScriptRoot "docker-compose.release.yml"
$envFile = Join-Path $PSScriptRoot ".env"
& docker compose -p psymas-desktop --env-file $envFile -f $compose down
