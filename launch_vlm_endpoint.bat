
@REM --n-predict 512 --host 0.0.0.0

@REM available quantizations q8_0 f16 q4_k_m

@REM call load_env.bat

start %LLAMA_SERVER_PATH% -hf %MODEL_NAME% ^
     --port %PORT% --ctx-size %CTX_SIZE%

@REM call pause


 