#creating new branch and push it to remote:
git switch -c feature/agent-profiles
git push -u origin feature/agent-profiles


# 1) Create virtual environment
python -m venv .venv

# 2) Activate it
.\.venv\Scripts\Activate.ps1

# 3) Upgrade pip (recommended)
python -m pip install --upgrade pip

# 4) Install dependencies
pip install -r requirements.txt




