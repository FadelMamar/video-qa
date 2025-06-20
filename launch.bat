call cd D:\workspace\repos\video-qa

call .\.venv\Scripts\activate

@REM load env variables from .env file
@REM if exist .env (
@REM     echo Loading .env file...
@REM     for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
@REM         REM Skip lines starting with # (comments)
@REM         echo %%a | findstr /r "^#" >nul
@REM         if errorlevel 1 (
@REM             REM Skip empty lines
@REM             if not "%%a"=="" (
@REM                 set "%%a=%%b"
@REM                 @REM echo Set %%a=%%b
@REM             )
@REM         )
@REM     )
@REM ) else (
@REM     echo .env file not found
@REM )

start streamlit run src/ui.py --server.port 8501

call D:\workspace\llama-cpp\llama-server.exe -hf ggml-org/Qwen2.5-VL-3B-Instruct-GGUF ^
     --port 8000 --n-predict 512 --host 0.0.0.0
