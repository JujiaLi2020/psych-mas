[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$AppRoot = $PSScriptRoot
$StartScript = Join-Path $AppRoot "Start-PsyMAS.ps1"
$StopScript = Join-Path $AppRoot "Stop-PsyMAS.ps1"
$MutexName = "Local\PsyMAS.Tray.v077"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$createdNew = $false
$mutex = New-Object System.Threading.Mutex($true, $MutexName, [ref]$createdNew)
if (-not $createdNew) {
    exit 0
}

function Invoke-PsyMASScript([string]$ScriptPath, [switch]$Wait) {
    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "PsyMAS control script not found: $ScriptPath"
    }
    $process = Start-Process powershell.exe -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath
    ) -WindowStyle Hidden -PassThru
    if ($Wait) { $process.WaitForExit() }
    return $process
}

function Get-PsyMASStatus {
    try {
        $running = & docker compose -p psymas-desktop ps --services --filter status=running 2>$null
        if ($LASTEXITCODE -eq 0 -and $running) { return "Running" }
        return "Stopped"
    } catch {
        return "Docker unavailable"
    }
}

$icon = New-Object System.Windows.Forms.NotifyIcon
$trayIconCandidates = @(
    (Join-Path $AppRoot "app-icon.ico"),
    (Join-Path $AppRoot "mark-navy.ico")
)
$trayIconPath = $trayIconCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($trayIconPath) {
    $icon.Icon = New-Object -TypeName System.Drawing.Icon -ArgumentList $trayIconPath
} else {
    $icon.Icon = [System.Drawing.SystemIcons]::Application
}
$icon.Text = "PsyMAS · checking status"
$icon.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenuStrip
$openItem = $menu.Items.Add("Open PsyMAS")
$startItem = $menu.Items.Add("Start service")
$stopItem = $menu.Items.Add("Stop service")
$restartItem = $menu.Items.Add("Restart service")
$menu.Items.Add("-") | Out-Null
$statusItem = $menu.Items.Add("Status: checking...")
$statusItem.Enabled = $false
$menu.Items.Add("-") | Out-Null
$exitItem = $menu.Items.Add("Exit controller")
$icon.ContextMenuStrip = $menu

$openItem.Add_Click({ Start-Process "http://localhost:8501" })
$startItem.Add_Click({
    try {
        Invoke-PsyMASScript $StartScript
        $icon.ShowBalloonTip(2500, "PsyMAS", "Starting the service...", [System.Windows.Forms.ToolTipIcon]::Info)
    } catch { $icon.ShowBalloonTip(3500, "PsyMAS", $_.Exception.Message, [System.Windows.Forms.ToolTipIcon]::Error) }
})
$stopItem.Add_Click({
    try {
        Invoke-PsyMASScript $StopScript
        $icon.ShowBalloonTip(2500, "PsyMAS", "The service is stopping. Your data is preserved.", [System.Windows.Forms.ToolTipIcon]::Info)
    } catch { $icon.ShowBalloonTip(3500, "PsyMAS", $_.Exception.Message, [System.Windows.Forms.ToolTipIcon]::Error) }
})
$restartItem.Add_Click({
    try {
        Invoke-PsyMASScript $StopScript
        Start-Sleep -Seconds 2
        Invoke-PsyMASScript $StartScript
        $icon.ShowBalloonTip(2500, "PsyMAS", "Restarting the service...", [System.Windows.Forms.ToolTipIcon]::Info)
    } catch { $icon.ShowBalloonTip(3500, "PsyMAS", $_.Exception.Message, [System.Windows.Forms.ToolTipIcon]::Error) }
})
$exitItem.Add_Click({
    try {
        # Exiting the controller is an explicit end-of-session action. Stop
        # the Compose project before releasing the notification resources.
        Invoke-PsyMASScript $StopScript -Wait | Out-Null
    } catch {
        $icon.ShowBalloonTip(3500, "PsyMAS", $_.Exception.Message, [System.Windows.Forms.ToolTipIcon]::Error)
    } finally {
        $icon.Visible = $false
        $icon.Dispose()
        $menu.Dispose()
        [System.Windows.Forms.Application]::ExitThread()
    }
})
$icon.Add_DoubleClick({ Start-Process "http://localhost:8501" })

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 10000
$timer.Add_Tick({
    $status = Get-PsyMASStatus
    $statusItem.Text = "Status: $status"
    $icon.Text = "PsyMAS · $status"
})
$timer.Start()
$statusItem.Text = "Status: $(Get-PsyMASStatus)"
$icon.Text = "PsyMAS · $($statusItem.Text -replace '^Status: ', '')"

[System.Windows.Forms.Application]::Run()
$timer.Stop()
$timer.Dispose()
$mutex.ReleaseMutex()
$mutex.Dispose()
