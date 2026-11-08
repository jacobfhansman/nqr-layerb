"""
To use: "python run_checks.py"

This creates a private virtual enviroment in .venv, installs the pinned
packages from requirements.txt and then runs the tests.
"""
import os
import subprocess
import sys
import venv
from pathlib import Path

minpython = (3,11)
root = Path(__file__).resolve().parent
venv_dir = root / ".venv"

#distinguish whether os is windows mac or linux
#to determine path to interpreter
def venv_python():
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python" #define virt env

def main():
    if sys.version_info < minpython: #if no python
        sys.exit(f"Python 3.11 or newer is required"
                 f"(found {sys.version.split()[0]}).")
        
    if not venv_python().exists():
        print(f"Creating virtual environment in {venv_dir}") #create virt env
        venv.create(venv_dir, with_pip=True)

    py = str(venv_python())
    print("Installing pinned packages from the requirements.txt") #install required packages
    subprocess.run([py, "-m", "pip", "install", "--quiet", "-r", str(root / "requirements.txt")], check=True)

    print("Running checks")
    #Returned 0 means pass every test, anything else returned means something failed
    result = subprocess.run([py, "-m", "pytest", "-v"], cwd=root)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()