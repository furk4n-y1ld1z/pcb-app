@echo off
echo Starting AOI Dashboard...

IF NOT EXIST ".venv" (
    echo First time setup: Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) ELSE (
    call .venv\Scripts\activate.bat
)

echo Launching Streamlit...
streamlit run app.py
pause