$ErrorActionPreference = "Stop"
$ComposeFile = Join-Path $PSScriptRoot "docker-compose.release.yml"
$EnvFile = Join-Path $PSScriptRoot ".env"
$AppDataRoot = Join-Path $env:LOCALAPPDATA "PsyMAS"
$LogDirectory = Join-Path $AppDataRoot "logs"
$LogFile = Join-Path $LogDirectory "start.log"
$transcriptStarted = $false

function Resolve-DockerExecutable {
    $command = Get-Command docker -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $candidates = @(
        "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
        "$env:LOCALAPPDATA\Programs\Docker\Docker\resources\bin\docker.exe"
    )
    return $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Show-StartupError([string]$Message) {
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show(
            "$Message`n`nDetails were saved to:`n$LogFile",
            "PsyMAS could not start",
            "OK",
            "Error"
        ) | Out-Null
    } catch {
        Write-Host "`n$Message" -ForegroundColor Red
        Write-Host "Details: $LogFile"
        Read-Host "Press Enter to close"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    Start-Transcript -Path $LogFile -Append | Out-Null
    $transcriptStarted = $true

    if (-not (Test-Path -LiteralPath $ComposeFile)) {
        throw "The PsyMAS service definition is missing. Reinstall PsyMAS."
    }
    if (-not (Test-Path -LiteralPath $EnvFile)) {
        throw "PsyMAS has not been configured. Open 'Configure PsyMAS AI' from the Start menu, then try again."
    }

    $DockerExe = Resolve-DockerExecutable
    if (-not $DockerExe) {
        throw "Docker Desktop is not installed. Install Docker Desktop, then try again."
    }

    & $DockerExe info *> $null
    if ($LASTEXITCODE -ne 0) {
        $desktopCandidates = @(
            "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
            "$env:LOCALAPPDATA\Programs\Docker\Docker\Docker Desktop.exe"
        )
        $desktop = $desktopCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($desktop) { Start-Process $desktop }

        $deadline = (Get-Date).AddMinutes(4)
        do {
            Start-Sleep -Seconds 4
            & $DockerExe info *> $null
            if ($LASTEXITCODE -eq 0) { break }
        } while ((Get-Date) -lt $deadline)
        if ($LASTEXITCODE -ne 0) {
            throw "Docker Desktop did not become ready. Open Docker Desktop and complete its first-run setup."
        }
    }

    & $DockerExe compose -p psymas-desktop --env-file $EnvFile -f $ComposeFile up -d
    if ($LASTEXITCODE -ne 0) {
        throw "The PsyMAS containers could not be started."
    }

    $deadline = (Get-Date).AddMinutes(3)
    $ready = $false
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 4
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch {
            Start-Sleep -Seconds 3
        }
    } while ((Get-Date) -lt $deadline)

    if (-not $ready) {
        throw "PsyMAS started, but the interface did not become ready within three minutes."
    }

    Stop-Transcript | Out-Null
    $transcriptStarted = $false
    Start-Process "http://localhost:8501"
} catch {
    $message = $_.Exception.Message
    Write-Error $message -ErrorAction Continue
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
        $transcriptStarted = $false
    }
    Show-StartupError $message
    exit 1
}
