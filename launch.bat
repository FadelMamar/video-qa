call cd D:\workspace\repos\video-qa

call .\.venv\Scripts\activate

call load_env.bat

start streamlit run app/ui.py --server.port 8501

@REM --n-predict 512 --host 0.0.0.0

@REM available quantizations q8_0 f16 q4_k_m

start D:\workspace\llama-cpp\llama-server.exe -hf ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m ^
     --port 8000 --ctx-size 20000

@REM call D:\workspace\llama-cpp\llama-server.exe -hf ggml-org/InternVL3-2B-Instruct-GGUF:F16 ^
@REM      --port 8000 
