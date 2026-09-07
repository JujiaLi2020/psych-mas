$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { throw "Run PsyMAS Setup before configuring AI support." }

$values = [ordered]@{}
foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^([^#=]+)=(.*)$') { $values[$matches[1].Trim()] = $matches[2] }
}

Write-Host "1. OpenRouter API (recommended)"
Write-Host "2. No AI"
Write-Host "3. Local Ollama (advanced)"
$choice = Read-Host "Choose 1, 2, or 3"
switch ($choice) {
    "1" {
        $values.OPENROUTER_API_KEY = (Read-Host "OpenRouter API key").Trim()
        $values.Remove("OLLAMA_CHAT_URL")
    }
    "3" {
        $values.OPENROUTER_API_KEY = ""
        $values.OLLAMA_CHAT_URL = "http://host.docker.internal:11434/api/chat"
    }
    default {
        $values.OPENROUTER_API_KEY = ""
        $values.Remove("OLLAMA_CHAT_URL")
    }
}
$envLines = @($values.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" })
[System.IO.File]::WriteAllLines($envFile, $envLines, (New-Object System.Text.UTF8Encoding($false)))

$compose = Join-Path $PSScriptRoot "docker-compose.release.yml"
& docker compose --env-file $envFile -f $compose up -d --force-recreate ui
Write-Host "PsyMAS AI configuration updated." -ForegroundColor Green
