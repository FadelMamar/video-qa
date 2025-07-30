
@REM --n-predict 512 --host 0.0.0.0

@REM available quantizations q8_0 f16 q4_k_m

call load_env.bat

start %LLAMA_SERVER_PATH% -hf %MODEL_NAME% ^
     --port %VLM_PORT% --ctx-size %CTX_SIZE%

call pause

@REM start D:\workspace\llama-cpp\llama-server.exe -hf ggml-org/Qwen3-1.7B-GGUF:q8_0 ^
@REM      --port 8001 

 