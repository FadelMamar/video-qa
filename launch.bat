call cd D:\workspace\repos\video-qa

call .\.venv\Scripts\activate

call load_env.bat

start streamlit run app/ui.py --server.port 8501

:: start launch_vlm_endpoint.bat
