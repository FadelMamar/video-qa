call cd /d %~dp0

call .\.venv\Scripts\activate

call load_env.bat

call streamlit run app/ui.py --server.port 8501

pause

pause

:: start launch_vlm_endpoint.bat
