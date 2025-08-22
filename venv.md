# Python Virtual Environment (venv) Guide

This guide explains how to create and use Python virtual environments using the built-in `venv` module. Virtual environments help isolate project dependencies, ensuring each project has its own packages without affecting the global Python installation.  

**Prerequisites:** Python 3.3+ and `pip`. Check your Python version with `python --version` or `python3 --version`.  

**Creating a virtual environment:**  
`python3 -m venv venv_nlp` (Linux/macOS) or `python -m venv venv_nlp` (Windows). 

**Activating the virtual environment:**  
Linux/macOS: `source venv_nlp/bin/activate`  
Windows (Command Prompt): `venv_nlp\Scripts\activate.bat`  
Windows (PowerShell): `venv_nlp\Scripts\Activate.ps1`  
Your shell prompt will show `(venv_nlp)` when activated.  

**Installing packages:** Use `pip install package_name`. Save installed packages with `pip freeze > requirements.txt` and install from a list with `pip install -r requirements.txt`.  

**Deactivating:** Run `deactivate` to exit the virtual environment.  

**Removing a virtual environment:** Delete the folder: `rm -rf venv_nlp` (Linux/macOS) or `rmdir /s /q venv_nlp` (Windows).  

**Tips:** Use `.venv` as the folder name for IDE detection, always activate before running scripts, and keep `requirements.txt` updated for reproducibility.
