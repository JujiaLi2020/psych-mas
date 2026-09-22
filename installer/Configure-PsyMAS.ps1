$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { throw "Run PsyMAS Setup before configuring AI support." }

$values = [ordered]@{}
foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^([^#=]+)=(.*)$') { $values[$matches[1].Trim()] = $matches[2] }
}

Write-Host "1. OpenRouter API (recommended)"
Write-Host "2. No AI"
Write-Host "3. Local Ollama (install if needed)"
$choice = Read-Host "Choose 1, 2, or 3"
switch ($choice) {
    "1" {
        $values.PSYMAS_LLM_PROVIDER = "openrouter"
        $values.PSYMAS_OPENROUTER_MODEL_ID = "openai/gpt-4o-mini"
        $values.OPENROUTER_API_KEY = (Read-Host "OpenRouter API key").Trim()
        $values.Remove("OLLAMA_CHAT_URL")
    }
    "3" {
        $values.OPENROUTER_API_KEY = ""
        $values.PSYMAS_LLM_PROVIDER = "local_ollama"
        $values.PSYMAS_OLLAMA_MODEL_ID = "llama3.1:8b"
        $values.OLLAMA_CHAT_URL = "http://host.docker.internal:11434/api/chat"
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            $install = Read-Host "Ollama is not installed. Install it with winget? [Y/n]"
            if (-not $install -or $install -match '^[Yy]') {
                if (Get-Command winget -ErrorAction SilentlyContinue) {
                    & winget install --exact --id Ollama.Ollama --silent --disable-interactivity --accept-package-agreements --accept-source-agreements
                    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
                }
            }
        }
        if (Get-Command ollama -ErrorAction SilentlyContinue) {
            & ollama pull $values.PSYMAS_OLLAMA_MODEL_ID
            Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        } else {
            Write-Warning "Ollama was not installed. Run 'ollama pull llama3.1:8b' later."
        }
    }
    default {
        $values.OPENROUTER_API_KEY = ""
        $values.PSYMAS_LLM_PROVIDER = "openrouter"
        $values.Remove("OLLAMA_CHAT_URL")
    }
}
$envLines = @($values.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" })
[System.IO.File]::WriteAllLines($envFile, $envLines, (New-Object System.Text.UTF8Encoding($false)))

$compose = Join-Path $PSScriptRoot "docker-compose.release.yml"
& docker compose --env-file $envFile -f $compose up -d --force-recreate ui
Write-Host "PsyMAS AI configuration updated." -ForegroundColor Green
