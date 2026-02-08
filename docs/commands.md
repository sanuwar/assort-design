# Assort Design — Copy/Paste Command Guide

> Quick reference for all common development tasks. Commands are grouped and ready to paste.

---

## 1. Git Workflow (Branching)

### Create a feature branch and push

```powershell
git switch -c feature/branch-name
git push -u origin feature/branch-name
```

---

## 2. Local Setup (Windows)

### Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### VS Code: Select Python Interpreter

1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Pick: `C:\Users\sanuw\AsortDesing\.venv\Scripts\python.exe`
3. Press `Ctrl+Shift+P` → **Developer: Reload Window**

---

## 3. Run Server (Local)

```powershell
python -m uvicorn app.main:app --reload --env-file .env
```

---

## 4. LangSmith Tracing (Optional)

Add to `.env`, then restart the server:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=asort-design
```

---

## 5. Mermaid Diagrams

### Offline Mermaid Note
Replace `static/mermaid.min.js` with the official Mermaid build to render mind maps without relying on the CDN fallback.

### Generate Mermaid output to docs/graph.md

```powershell
python -m app.graph --mermaid | Out-File -Encoding utf8 docs/graph.md
```

---

## 6. Database (SQLite)

### Reset local DB (after schema changes)

```powershell
Remove-Item -Force .\data\asort_design.db
```

### List tables (inspect DB)

```powershell
python -c "import sqlite3; conn=sqlite3.connect('data/asort_design.db'); print(conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())"
```

---

## 7. VPS Access (DigitalOcean)

### SSH into VPS (Windows CMD)

```bat
cd c:/
ssh root@157.245.8.184
```

### Navigate on VPS

```bash
# Go to deploy folder
cd /opt/assort-design

# Check current folder
pwd

# Go to filesystem root
cd /
```

---

## 8. VPS Deploy/Update Routine

### Pull latest image and restart services (on VPS)

```bash
cd /opt/assort-design && docker compose pull && docker compose up -d
```

---

## Quick Reference Summary

| Task | Command |
|------|---------|
| **Create feature branch** | `git switch -c feature/name` |
| **Activate venv** | `.\.venv\Scripts\Activate.ps1` |
| **Run server locally** | `python -m uvicorn app.main:app --reload --env-file .env` |
| **Reset database** | `Remove-Item -Force .\data\asort_design.db` |
| **SSH to VPS** | `ssh root@157.245.8.184` |
| **Deploy on VPS** | `cd /opt/assort-design && docker compose pull && docker compose up -d` |

---

## Notes

- Always activate the virtual environment before running Python commands
- Use `--reload` flag for development to auto-restart on code changes
- Remember to restart the server after modifying `.env` file
- VPS commands should be run after SSH-ing into the server
