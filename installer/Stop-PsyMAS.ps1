$ErrorActionPreference = "Stop"
$compose = Join-Path $PSScriptRoot "docker-compose.release.yml"
$envFile = Join-Path $PSScriptRoot ".env"
& docker compose --env-file $envFile -f $compose down
