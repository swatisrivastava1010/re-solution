# Re:Solution installer for Windows.
#
# Paste this one line into PowerShell and press Enter:
#   irm https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.ps1 | iex
#
# It installs Re:Solution into %LOCALAPPDATA%\Re-Solution, adds Start Menu and Desktop
# shortcuts, and starts it. Run the same line again to update (close Re:Solution first):
# your settings, keys, downloaded models and pictures are kept. It never needs
# administrator rights, and is meant for a normal PowerShell window, not "Run as
# administrator" (that would install it for the administrator's account).
#
# Options, when run as a file (powershell -ExecutionPolicy Bypass -File install.ps1 ...):
#   -Dir <folder>      install somewhere else
#   -Source <folder>   copy the app from a folder on this PC instead of GitHub
#   -NoShortcuts       don't add Start Menu or Desktop shortcuts
#   -NoStart           don't start Re:Solution at the end
#   -DryRun            only check this PC and say what would happen
#   -FakeGpu <value>   with -DryRun, for testing: pretend nvidia-smi reported this driver
#                      version and compute capability (for example 590.12,8.6, or
#                      581.57,6.1 for a GTX 1080), or "none" for no NVIDIA card
#
# The PyTorch build is chosen with the same rules as install.sh and the Start Re-Solution
# launchers: an NVIDIA card of the GeForce RTX 20 / GTX 16 generation or newer (compute
# capability 7.5) with driver 580 or newer gets "cu130"; anything else "cpu".
#
# For a fork (or testing), $env:RE_SOLUTION_REPO_URL and $env:RE_SOLUTION_ZIP_URL change
# where the app is downloaded from, and $env:RE_SOLUTION_PORT where it looks for a running
# Re:Solution.
#
# This script never calls "exit" when pasted (irm | iex): that would close the window
# before the message could be read.

[CmdletBinding()]
param(
    [string]$Dir = "",
    [string]$Source = "",
    [switch]$NoShortcuts,
    [switch]$NoStart,
    [switch]$DryRun,
    [string]$FakeGpu = ""
)

$ReSolutionRepoUrl = if ($env:RE_SOLUTION_REPO_URL) { $env:RE_SOLUTION_REPO_URL } else { 'https://github.com/swatisrivastava1010/re-solution' }
$ReSolutionBranch = 'main'
$ReSolutionZipUrl = if ($env:RE_SOLUTION_ZIP_URL) { $env:RE_SOLUTION_ZIP_URL } else { "$ReSolutionRepoUrl/archive/refs/heads/$ReSolutionBranch.zip" }
$ReSolutionMinCudaDriver = 580   # the cu130 PyTorch build needs NVIDIA driver 580 or newer,
$ReSolutionMinComputeCap = 75    # and a card it has code for: compute capability 7.5 (RTX 20 / GTX 16) or newer
$ReSolutionPort = 7860           # where a running Re:Solution answers
if ("$env:RE_SOLUTION_PORT" -match '^\d+$') { $ReSolutionPort = [int]$env:RE_SOLUTION_PORT }
$ReSolutionSteps = 5

# --- messages ----------------------------------------------------------------------------

function Write-Step([int]$Number, [string]$Text) {
    Write-Host ""
    Write-Host "$Number/$ReSolutionSteps $Text" -ForegroundColor Cyan
}
function Write-Ok([string]$Text) { Write-Host "    OK  $Text" -ForegroundColor Green }
function Write-Info([string]$Text) { Write-Host "    $Text" }
function Write-Warn([string]$Text) { Write-Host "    !   $Text" -ForegroundColor Yellow }

# Stops the install with a friendly explanation (shown by the code at the bottom).
function Stop-Install([string]$Problem, [string[]]$Hints = @()) {
    $exception = New-Object System.Exception ((@("Sorry, $Problem") + $Hints) -join [Environment]::NewLine)
    $exception.Data['Friendly'] = $true
    throw $exception
}

# --- choosing the PyTorch build (the same rules as install.sh and the launchers) ---------

# The first graphics card's answer to one nvidia-smi question (for example
# 'driver_version'), or $null without a working NVIDIA driver.
function Get-NvidiaSmiAnswer([string]$Query) {
    $ErrorActionPreference = 'Continue'   # nvidia-smi may print warnings; they're not fatal
    $command = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
        $exe = $command.Path
    } elseif ($env:ProgramFiles) {
        # Older drivers put it here instead of on the PATH.
        $exe = Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    } else {
        return $null
    }
    if (-not (Test-Path -LiteralPath $exe)) { return $null }
    try {
        $output = & $exe "--query-gpu=$Query" --format=csv,noheader 2>$null
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0) { return $null }
    return "$(@($output)[0])"   # the first graphics card
}

# The NVIDIA driver's major version (for example 590), or $null without a working driver.
# $Fake (for testing) stands in for nvidia-smi's answers, for example '581.57,6.1'.
function Get-NvidiaDriverMajor([string]$Fake) {
    $text = if ($Fake) { @($Fake -split '[,\s]+')[0] } else { Get-NvidiaSmiAnswer 'driver_version' }
    if ("$text" -match '^\s*(\d+)(\.|\s*$)') { return [int]$Matches[1] }
    return $null
}

# The first card's compute capability (its generation, as the graphics software counts
# it) as a whole number: 61 for a GTX 1080's 6.1, 86 for an RTX 3060's 8.6. $null when it
# can't be read. Drivers older than 510 don't know the question.
function Get-NvidiaComputeCap([string]$Fake) {
    $text = if ($Fake) { @($Fake -split '[,\s]+')[1] } else { Get-NvidiaSmiAnswer 'compute_cap' }
    if ("$text" -match '^\s*(\d{1,2})\.(\d)\s*$') { return 10 * [int]$Matches[1] + [int]$Matches[2] }
    return $null
}

# The cu130 build needs driver 580 and a card it has code for. Driver 580 is also the last
# one for GTX 10 cards, so a new driver alone doesn't say the card is new enough.
function Get-TorchVariant($DriverMajor, $ComputeCap) {
    if ($null -ne $DriverMajor -and $DriverMajor -ge $ReSolutionMinCudaDriver -and
        $null -ne $ComputeCap -and $ComputeCap -ge $ReSolutionMinComputeCap) { return 'cu130' }
    return 'cpu'
}

# --- small helpers ---------------------------------------------------------------------

# An absolute path without a trailing slash (the folder may not exist yet).
function Resolve-FullPath([string]$Path) {
    $expanded = [Environment]::ExpandEnvironmentVariables($Path)
    if ($expanded -eq '~' -or $expanded.StartsWith('~\') -or $expanded.StartsWith('~/')) {
        $expanded = $HOME + $expanded.Substring(1)
    }
    $full = [IO.Path]::GetFullPath([IO.Path]::Combine((Get-Location).ProviderPath, $expanded))
    $root = [IO.Path]::GetPathRoot($full)
    if ($full.Length -gt $root.Length) { $full = $full.TrimEnd('\', '/') }
    return $full
}

function Test-LooksLikeApp([string]$Path) {
    return (Test-Path -LiteralPath (Join-Path $Path 'pyproject.toml') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path 'upscaler') -PathType Container)
}

function Test-EmptyFolder([string]$Path) {
    return (Test-Path -LiteralPath $Path -PathType Container) -and
        -not (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)
}

# Windows on Arm (for example a Snapdragon laptop). This PowerShell may be an Intel version
# running under emulation, which reports AMD64, so Windows' own setting is checked too.
function Test-WindowsOnArm {
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { return $true }
    try {
        $key = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' `
            -Name PROCESSOR_ARCHITECTURE -ErrorAction Stop
        return $key.PROCESSOR_ARCHITECTURE -eq 'ARM64'
    } catch {
        return $false   # not Windows, or the setting can't be read: assume an Intel or AMD PC
    }
}

# True in a window opened with "Run as administrator". Then the folders this installer uses
# (LOCALAPPDATA, the Start Menu, the Desktop) can belong to the administrator who typed
# their password, not to the person who wants Re:Solution. With UAC switched off, every
# program an administrator starts runs this way, so that isn't counted.
function Test-RunAsAdministrator {
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal $identity
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { return $false }
        $uac = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System' `
            -Name EnableLUA -ErrorAction Stop
        return $uac.EnableLUA -ne 0
    } catch {
        return $false   # not Windows, or the setting can't be read
    }
}

# True when Re:Solution is already running on this PC: it answers on its port the way the
# app itself checks (cli._probe_running). Never through a proxy: the PC's proxy settings
# could send even 127.0.0.1 there.
function Test-AppRunning([int]$Port) {
    try {
        $request = [Net.WebRequest]::Create("http://127.0.0.1:$Port/api/status")
        $request.Proxy = $null
        $request.Timeout = 5000
        $response = $request.GetResponse()
        try {
            $reader = New-Object IO.StreamReader ($response.GetResponseStream())
            return $reader.ReadToEnd().Contains('"is_host"')
        } finally {
            $response.Close()
        }
    } catch {
        return $false   # nothing there, or some other program
    }
}

function Test-Inside([string]$Path, [string]$Folder) {
    $separator = [IO.Path]::DirectorySeparatorChar
    return $Path.Equals($Folder, [StringComparison]::OrdinalIgnoreCase) -or
        $Path.StartsWith($Folder.TrimEnd($separator) + $separator, [StringComparison]::OrdinalIgnoreCase)
}

# Copies the app's files from $From into $To. This PC's own files in $To (the installed
# packages in .venv, .torch-variant, test pictures in samples) are left alone.
function Copy-AppFiles([string]$From, [string]$To) {
    New-Item -ItemType Directory -Force -Path $To | Out-Null
    # robocopy skips a bare name at any depth, so the top-level things are given as full
    # paths: a bare "samples" would also skip the bundled upscaler\web\samples. Bare names
    # are only for clutter that can be skipped wherever it is.
    $topLevelFolders = '.git', '.venv', 'samples', 'Re-Solution.app' | ForEach-Object { Join-Path $From $_ }
    $torchVariant = Join-Path $From '.torch-variant'
    # /R:2 /W:1: don't wait for ages on a locked file (robocopy's default is days).
    # (An array given to a program becomes one argument per item.)
    & robocopy $From $To /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP `
        /XD $topLevelFolders __pycache__ .pytest_cache .ruff_cache `
        /XF $torchVariant *.part .DS_Store | Out-Null
    if ($LASTEXITCODE -ge 8) {   # robocopy: 0-7 mean success
        Stop-Install "couldn't copy the app into ""$To""." @('If Re:Solution is open, close its window, then run the same command again.')
    }
    $readme = Join-Path (Join-Path $From 'samples') 'README.md'
    if (Test-Path -LiteralPath $readme) {
        $samples = Join-Path $To 'samples'
        New-Item -ItemType Directory -Force -Path $samples | Out-Null
        Copy-Item -LiteralPath $readme -Destination (Join-Path $samples 'README.md') -Force
    }
}

# Runs uv with an activated virtual environment (if any) hidden, so it can't redirect the install.
function Invoke-Uv([string]$Uv, [string[]]$Arguments) {
    $saved = @{}
    foreach ($name in 'VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT') {
        $saved[$name] = [Environment]::GetEnvironmentVariable($name)
        [Environment]::SetEnvironmentVariable($name, $null)
    }
    try {
        & $Uv @Arguments | Out-Host
        return $LASTEXITCODE
    } finally {
        foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
    }
}

# --- the install -----------------------------------------------------------------------

function Install-ReSolution {
    param([string]$Dir, [string]$Source, [bool]$NoShortcuts, [bool]$NoStart, [bool]$DryRun, [string]$FakeGpu)

    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'   # Windows PowerShell's progress bar slows downloads a lot
    $offline = @('Check that this PC is connected to the internet, then run the same command again.')
    $onWindows = ($PSVersionTable.PSEdition -ne 'Core') -or $IsWindows

    if ($FakeGpu -and -not $DryRun) { Stop-Install '-FakeGpu is only for testing, together with -DryRun.' }
    if (-not $Dir) {
        $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { [Environment]::GetFolderPath('LocalApplicationData') }
        $Dir = Join-Path $localAppData 'Re-Solution'
    }
    $Dir = Resolve-FullPath $Dir
    if ($Source) {
        $Source = Resolve-FullPath $Source
        if (-not (Test-LooksLikeApp $Source)) {
            Stop-Install """$Source"" doesn't look like a Re:Solution folder." @('It should contain pyproject.toml and the upscaler folder.')
        }
        if (Test-Inside $Dir $Source) {
            Stop-Install "the install folder can't be the source folder or inside it." @('Choose another folder with -Dir.')
        }
    }

    Write-Host 'Installing Re:Solution, the free picture and video upscaler.' -ForegroundColor White
    if ($DryRun) { Write-Host 'Dry run: checking only, nothing will be changed.' }

    # 1. This PC and its graphics card ------------------------------------------------
    Write-Step 1 'Checking your computer...'
    if (-not $onWindows -and -not $DryRun) {
        Stop-Install 'this installer is for Windows.' @('On a Mac, paste this into Terminal instead:',
            '  curl -fsSL https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.sh | bash')
    }
    if ($onWindows -and -not [Environment]::Is64BitOperatingSystem) {
        Stop-Install 'Re:Solution needs 64-bit Windows.'
    }
    if (Test-WindowsOnArm) {
        Stop-Install 'Re:Solution needs a Windows PC with an Intel or AMD processor.' @(
            'PyTorch, the engine it runs on, has no complete version for Windows on Arm yet.')
    }
    if ($onWindows -and (Test-RunAsAdministrator)) {
        $hints = @('Re:Solution installs just for you and needs no administrator rights.',
            'Open PowerShell the normal way (not "Run as administrator"), then paste the same command again.')
        if (-not $DryRun) { Stop-Install 'this PowerShell window was opened with "Run as administrator".' $hints }
        Write-Warn 'This PowerShell window was opened with "Run as administrator".'
        foreach ($hint in $hints) { Write-Info $hint }
    }
    $driver = Get-NvidiaDriverMajor $FakeGpu
    $newDriver = $null -ne $driver -and $driver -ge $ReSolutionMinCudaDriver
    # Only asked with a new driver: older ones don't know the question.
    $computeCap = if ($newDriver) { Get-NvidiaComputeCap $FakeGpu } else { $null }
    $variant = Get-TorchVariant $driver $computeCap
    if ($variant -eq 'cu130') {
        Write-Ok "NVIDIA graphics card found (driver $driver): Re:Solution will use it"
    } elseif ($newDriver -and $null -ne $computeCap) {
        Write-Warn 'Your NVIDIA graphics card is too old for the graphics software Re:Solution uses, so it will use the processor instead (CPU only: slow).'
        Write-Info 'Re:Solution can use a GeForce RTX 20 or GTX 16 series card, or newer.'
    } elseif ($null -ne $driver -and -not $newDriver) {
        Write-Warn "Your NVIDIA driver ($driver) is older than version $ReSolutionMinCudaDriver, so Re:Solution will use the processor instead (CPU only: slow)."
        Write-Info 'For much faster upscaling, update the driver from https://www.nvidia.com/drivers and run this installer again.'
    } else {
        Write-Warn 'No NVIDIA graphics card found, so Re:Solution will use the processor (CPU only: slow).'
    }
    Write-Info "PyTorch build: $variant"
    # Updating under a running copy would swap its files while it uses them, and the start
    # at the end would only open the old copy again.
    if (Test-AppRunning $ReSolutionPort) {
        if (-not $DryRun) {
            Stop-Install "Re:Solution is open, so it can't be installed or updated right now." @(
                'Close its window (or press Ctrl+C in it), then run the same command again. Nothing was changed.')
        }
        Write-Warn 'Re:Solution is open. Close its window before installing for real.'
    }
    try {
        $freeGb = [math]::Floor((New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($Dir))).AvailableFreeSpace / 1GB)
        if ($freeGb -lt 10) { Write-Warn "Only $freeGb GB of free space left. Re:Solution needs a few GB, and Photo-real 7 GB more." }
    } catch {
        # Free space is only a friendly warning.
    }

    # 2. uv ---------------------------------------------------------------------------
    Write-Step 2 'Getting uv, the small tool that installs everything else...'
    $uvBinDir = Join-Path (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local') 'bin'
    $env:Path = $uvBinDir + [IO.Path]::PathSeparator + $env:Path   # this window only
    $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $uv = $null
    if ($uvCommand) {
        $uv = $uvCommand.Path
        Write-Ok 'uv is already installed'
        # Keep a uv this installer put there up to date, so it can read the newest uv.lock.
        if (-not $DryRun -and $uv -eq (Join-Path $uvBinDir 'uv.exe')) {
            try { & $uv self update --quiet *> $null } catch { }
        }
    } elseif ($DryRun) {
        Write-Info "Would install uv into $uvBinDir"
    } else {
        Write-Info "Installing uv into $uvBinDir (your Windows settings stay as they are)..."
        $saved = @{ UV_NO_MODIFY_PATH = $env:UV_NO_MODIFY_PATH; UV_INSTALL_DIR = $env:UV_INSTALL_DIR }
        try {
            $env:UV_NO_MODIFY_PATH = '1'
            $env:UV_INSTALL_DIR = $uvBinDir
            $shellExe = if ($PSVersionTable.PSEdition -eq 'Core') { Join-Path $PSHOME 'pwsh.exe' } else { Join-Path $PSHOME 'powershell.exe' }
            & $shellExe -NoProfile -ExecutionPolicy ByPass -Command 'irm https://astral.sh/uv/install.ps1 | iex' | Out-Null
        } finally {
            $env:UV_NO_MODIFY_PATH = $saved.UV_NO_MODIFY_PATH
            $env:UV_INSTALL_DIR = $saved.UV_INSTALL_DIR
        }
        $uv = Join-Path $uvBinDir 'uv.exe'
        if (-not (Test-Path -LiteralPath $uv)) { Stop-Install "couldn't install uv." $offline }
        Write-Ok 'uv installed'
    }

    # 3. The app itself ---------------------------------------------------------------
    Write-Step 3 'Downloading Re:Solution...'
    $installed = Test-LooksLikeApp $Dir
    if ((Test-Path -LiteralPath $Dir) -and -not $installed -and -not (Test-EmptyFolder $Dir)) {
        Stop-Install "the folder ""$Dir"" already has other things in it." @('Re:Solution needs a folder of its own. Choose another one with -Dir, or empty this one.')
    }
    $git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($DryRun) {
        if ($Source) { Write-Info "Would copy the app from $Source into $Dir" }
        elseif ($installed) { Write-Info "Would update the copy in $Dir to the latest version" }
        else { Write-Info "Would download the app from $ReSolutionRepoUrl into $Dir" }
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dir) | Out-Null
        $temp = Join-Path ([IO.Path]::GetTempPath()) ('re-solution-install-' + [guid]::NewGuid().ToString('N'))
        try {
            if ($Source) {
                Copy-AppFiles $Source $Dir
                Write-Ok "Copied from $Source"
            } elseif ($git -and (Test-Path -LiteralPath (Join-Path $Dir '.git'))) {
                # fetch + reset (rather than pull) can't get stuck on old local edits.
                & $git.Path -C $Dir fetch --quiet --depth 1 origin $ReSolutionBranch
                if ($LASTEXITCODE -eq 0) { & $git.Path -C $Dir reset --quiet --hard FETCH_HEAD }
                if ($LASTEXITCODE -ne 0) { Stop-Install "couldn't download the update." $offline }
                Write-Ok 'Updated to the latest version'
            } elseif ($git -and -not $installed) {
                & $git.Path clone --quiet --depth 1 --branch $ReSolutionBranch $ReSolutionRepoUrl $Dir
                if ($LASTEXITCODE -ne 0) { Stop-Install "couldn't download Re:Solution." $offline }
                Write-Ok "Downloaded into $Dir"
            } else {
                # No git, or an earlier copy that wasn't made with git: fetch a fresh copy,
                # then copy it over the old one.
                New-Item -ItemType Directory -Force -Path $temp | Out-Null
                if ($git) {
                    $fresh = Join-Path $temp 'app'
                    & $git.Path clone --quiet --depth 1 --branch $ReSolutionBranch $ReSolutionRepoUrl $fresh
                    if ($LASTEXITCODE -ne 0) { Stop-Install "couldn't download Re:Solution." $offline }
                } else {
                    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
                    $zip = Join-Path $temp 'app.zip'
                    try {
                        Invoke-WebRequest -Uri $ReSolutionZipUrl -OutFile $zip -UseBasicParsing
                    } catch {
                        Stop-Install "couldn't download Re:Solution." $offline
                    }
                    Unblock-File -LiteralPath $zip
                    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $temp 'unzipped') -Force
                    $fresh = Get-ChildItem -LiteralPath (Join-Path $temp 'unzipped') -Directory | Select-Object -First 1 -ExpandProperty FullName
                    if (-not $fresh -or -not (Test-LooksLikeApp $fresh)) {
                        Stop-Install 'the download was damaged.' @('Please run the same command again.')
                    }
                }
                Copy-AppFiles $fresh $Dir
                if ($installed) { Write-Ok 'Updated to the latest version' } else { Write-Ok "Downloaded into $Dir" }
            }
        } finally {
            if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue }
        }
        [IO.File]::WriteAllText((Join-Path $Dir '.torch-variant'), "$variant`r`n")
    }

    # 4. The app's parts (Python, PyTorch, ...) -------------------------------------------
    $size = if ($variant -eq 'cu130') { 'about 3 GB' } else { 'about 400 MB' }
    Write-Step 4 "Installing the app's parts (the first time, this downloads $size and can take a while)..."
    if ($DryRun) {
        Write-Info "Would run: uv sync --frozen --no-dev --extra $variant"
    } else {
        $status = Invoke-Uv $uv @('sync', '--frozen', '--no-dev', '--extra', $variant, '--directory', $Dir)
        if ($status -ne 0) {
            Stop-Install "couldn't install the app's parts." @(
                'Check the internet connection and that there is free disk space, then run the same command again.',
                'If Re:Solution is open, close its window first.')
        }
        Write-Ok 'Everything is installed'
    }

    # 5. Shortcuts ----------------------------------------------------------------------
    Write-Step 5 'Creating shortcuts...'
    $launcher = Join-Path $Dir 'Start Re-Solution.bat'
    $folders = @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop')) | Where-Object { $_ }
    if ($NoShortcuts) {
        Write-Info 'Skipped: no Start Menu or Desktop shortcuts were added.'
    } elseif ($DryRun) {
        Write-Info 'Would add Re-Solution shortcuts to the Start Menu and the Desktop'
    } else {
        # Per-user Start Menu and Desktop folders: no administrator rights needed.
        $shell = New-Object -ComObject WScript.Shell
        foreach ($folder in $folders) {
            try {
                $shortcut = $shell.CreateShortcut((Join-Path $folder 'Re-Solution.lnk'))
                $shortcut.TargetPath = $launcher
                $shortcut.WorkingDirectory = $Dir
                $shortcut.Description = 'Start Re:Solution, the free picture and video upscaler'
                $shortcut.Save()
                Write-Ok "Added Re-Solution to $folder"
            } catch {
                Write-Warn "Couldn't add a shortcut to $folder. You can still start Re:Solution with ""Start Re-Solution"" in $Dir"
            }
        }
    }

    # Done ------------------------------------------------------------------------------
    Write-Host ''
    if ($DryRun) {
        Write-Host 'Dry run finished: nothing was changed.' -ForegroundColor White
        return
    }
    Write-Host "Re:Solution is installed in $Dir" -ForegroundColor Green
    Write-Info 'To update later, run the same command again. Your settings, keys, models and pictures are kept.'
    if ($NoStart) {
        if ($NoShortcuts) { Write-Info "Start it any time by double-clicking ""Start Re-Solution"" in $Dir" }
        else { Write-Info 'Start it any time with Re-Solution in the Start Menu or on the Desktop.' }
        return
    }
    Write-Host 'Starting Re:Solution in a new window...' -ForegroundColor White
    Write-Info 'Keep that window open while you use Re:Solution. Close it to stop.'
    Start-Process -FilePath $launcher -WorkingDirectory $Dir
}

$ReSolutionExitCode = 0
try {
    Install-ReSolution -Dir $Dir -Source $Source -NoShortcuts $NoShortcuts.IsPresent -NoStart $NoStart.IsPresent `
        -DryRun $DryRun.IsPresent -FakeGpu $FakeGpu
} catch {
    $ReSolutionExitCode = 1
    Write-Host ''
    if ($_.Exception.Data['Friendly']) {
        Write-Host $_.Exception.Message -ForegroundColor Red
    } else {
        Write-Host 'Sorry, something went wrong while installing Re:Solution:' -ForegroundColor Red
        Write-Host "  $($_.Exception.Message)"
        Write-Host 'Check that this PC is connected to the internet, then run the same command again.'
        Write-Host 'Your settings, keys and pictures are safe.'
    }
}
# Only a script run as a file may set an exit code; when pasted, "exit" would close the window.
if ($PSCommandPath) { exit $ReSolutionExitCode }
