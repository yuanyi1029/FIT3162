# Setup 

## Step 1: Install Python Virtual Environment
```
pip install virtualenv
```

## Step 2: Initialize a Virtual Environment 
```
python -m venv venv 
./venv/scripts/activate
```

## Step 3: Install Dependencies  
```
pip install -r requirements.txt 
cd mcunet 
pip install -e .
```

## Step 4: Run The File  
```
streamlit run main.py
```