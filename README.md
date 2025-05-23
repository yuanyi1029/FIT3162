# Setup Guide

## Step 1: Check and Install Python 3.10
Ensure Python 3.10 is installed on your system:
```bash
python3.10 --version
```

If not installed, you can install it via Homebrew (for macOS):
```bash
brew install python@3.10
```

## Step 2: Remove Any Existing Virtual Environment
If there is an existing venv folder, remove it to avoid conflicts:
```bash
rm -rf venv
```

## Step 3: Create and Activate a New Virtual Environment
Create a virtual environment using Python 3.10:
```bash
python3.10 -m venv venv
```

Activate the virtual environment:
```bash
source venv/bin/activate
```

## Step 4: Upgrade pip and Install Dependencies
Upgrade pip to the latest version:
```bash
pip install --upgrade pip
```

Install the local mcunet package in editable mode:
```bash
cd mcunet
pip install -e .
```

## Step 5: Run the Application
To launch the application using Streamlit:
```bash
streamlit run main.py
```