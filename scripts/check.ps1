$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:SOULDRIVE_PYTHON) { $env:SOULDRIVE_PYTHON } else { "D:\Anaconda\envs\souldrive\python.exe" }

Push-Location $Root
try {
    & $Python -m unittest discover -s tests -v
    & $Python -m compileall -q core tests

    Push-Location "souldrive-ui"
    try {
        npm.cmd run build

        Push-Location "src-tauri"
        try {
            cargo check
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
