# Environment Setup

This repo uses Git hooks to enforce that a Python virtual environment is active
and to keep `requirements.txt` in sync on every commit.

## Python Virtual Environment

Windows (PowerShell):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Enable Git Hooks

Windows:
```powershell
scripts\setup-hooks.ps1
```

Linux/macOS:
```bash
./scripts/setup-hooks.sh
```

Once enabled, the pre-commit hook will:
- Block commits if no virtual environment is active.
- Update and stage `requirements.txt` based on the active venv.
