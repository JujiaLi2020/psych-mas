$ErrorActionPreference = "Stop"
$composeCandidates = @(
    (Join-Path $PSScriptRoot "docker-compose.release.yml"),
    (Join-Path $PSScriptRoot "..\docker-compose.release.yml")
)
$compose = $composeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $compose) { throw "PsyMAS Compose file was not found." }

$envCandidates = @(
    (Join-Path $PSScriptRoot ".env"),
    (Join-Path $PSScriptRoot "..\.env")
)
$envFile = $envCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$composeArgs = @("compose", "-p", "psymas-desktop")
if ($envFile) { $composeArgs += @("--env-file", $envFile) }
$composeArgs += @("-f", $compose, "down")
& docker @composeArgs
if ($LASTEXITCODE -ne 0) { throw "PsyMAS services could not be stopped." }
exit 0
