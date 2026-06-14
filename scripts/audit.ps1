$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:SOULDRIVE_PYTHON) { $env:SOULDRIVE_PYTHON } else { "D:\Anaconda\envs\souldrive\python.exe" }
$Failed = $false

Push-Location $Root
try {
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { $Failed = $true }

    & $Python -m pip_audit --progress-spinner off
    if ($LASTEXITCODE -ne 0) { $Failed = $true }

    Push-Location "souldrive-ui"
    try {
        npm.cmd audit --omit=dev
        if ($LASTEXITCODE -ne 0) { $Failed = $true }

        Push-Location "src-tauri"
        try {
            if (Get-Command cargo-audit -ErrorAction SilentlyContinue) {
                cargo audit
                if ($LASTEXITCODE -ne 0) { $Failed = $true }
            } else {
                Write-Warning "cargo-audit is not installed; Rust advisory audit was skipped."
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}

if ($Failed) {
    exit 1
}
