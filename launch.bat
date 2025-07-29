call cd /d %~dp0

call .\.venv\Scripts\activate

call load_env.bat

start streamlit run app/ui.py --server.port 8501

pause

:: start launch_vlm_endpoint.bat
