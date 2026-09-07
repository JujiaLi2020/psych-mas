#define MyAppName "PsyMAS"
#define MyAppVersion "0.7.4"
#define MyAppPublisher "Jujia Li"
#define MyAppURL "https://github.com/JujiaLi2020/psych-mas"

[Setup]
AppId={{B1B0742B-383E-41B2-97BA-1BB29846C3DD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={localappdata}\Programs\PsyMAS
DefaultGroupName=PsyMAS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=PsyMAS-Setup-Windows-v{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=PsyMAS v{#MyAppVersion}
LicenseFile=..\LICENSE

[Files]
Source: "Install-PsyMAS.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "Start-PsyMAS.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "Stop-PsyMAS.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "Configure-PsyMAS.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docker-compose.release.yml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.env.example"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Start PsyMAS"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-PsyMAS.ps1"""; WorkingDir: "{app}"
Name: "{group}\Configure PsyMAS AI"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Configure-PsyMAS.ps1"""; WorkingDir: "{app}"
Name: "{group}\Stop PsyMAS"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Stop-PsyMAS.ps1"""; WorkingDir: "{app}"
Name: "{autodesktop}\PsyMAS"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-PsyMAS.ps1"""; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Install-PsyMAS.ps1"""; WorkingDir: "{app}"; Description: "Install Docker if needed and start PsyMAS"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Stop-PsyMAS.ps1"""; WorkingDir: "{app}"; Flags: runhidden skipifdoesntexist; RunOnceId: "StopPsyMAS"
