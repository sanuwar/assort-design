# Commands - Asort Design

## Branching
```powershell
git switch -c feature/agent-profiles
git push -u origin feature/agent-profiles
```

## Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## VS Code Interpreter
Ctrl+Shift+P → Python: Select Interpreter  
Pick: `C:\Users\sanuw\AsortDesing\.venv\Scripts\python.exe`

Ctrl+Shift+P → Developer: Reload Window

## Run Server
```powershell
python -m uvicorn app.main:app --reload --env-file .env
```

## LangSmith Tracing
Set these in `.env`, then restart the server:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=asort-design
```

## Mermaid (Offline Rendering)
Replace `static/mermaid.min.js` with the official Mermaid build to render mind maps
without relying on the CDN fallback.

## Reset Local DB (After Schema Changes)
```powershell
Remove-Item -Force .\data\asort_design.db
```
