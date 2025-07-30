import fire
import subprocess
import os
from pathlib import Path
from watcher.analyzer.analysis import analyze_video
from watcher.config import PredictionConfig
from typing import Union, Optional
import json
from dotenv import load_dotenv


def analyze(video: str, args: dict, metadata: Optional[dict] = None,activity_analysis: bool = False) -> dict:
    args = PredictionConfig(**args)
    result = analyze_video(video=video, args=args, metadata=metadata,activity_analysis=activity_analysis)
    return json.dumps(vars(result))


def launch_vlm(llama_server_path: str=None,model_name: str=None,port: int = 8000,ctx_size: int = 20000,load_env: bool = True):
    
    
    # Path to the batch file
    cwd = Path(__file__).resolve().parent
    
    if load_env:
        load_dotenv(cwd / ".env",override=True)

    if llama_server_path:
        assert os.path.exists(llama_server_path), f"llama.cpp server file not found: {llama_server_path}"
        os.environ["LLAMA_SERVER_PATH"] = llama_server_path
    if model_name:
        assert model_name.startswith("ggml-org/"), f"model_name must start with 'ggml-org/': {model_name}"
        os.environ["MODEL_NAME"] = model_name
    if port:
        assert isinstance(port, int), f"port must be an integer: {port}"
        os.environ["PORT"] = str(port)
    if ctx_size:
        assert isinstance(ctx_size, int), f"ctx_size must be an integer: {ctx_size}"
        os.environ["CTX_SIZE"] = str(ctx_size)
    
    # Launch the batch file as a subprocess in the background
    llama_server_log = os.environ.get("LLAMA_SERVER_LOG","llama_server.log")
    with open(llama_server_log, "w") as f:
        subprocess.Popen(
            "launch_vlm_endpoint.bat",
            shell=False,
            cwd=cwd,
            #creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=os.environ.copy(),
            stdout=f,
            stderr=f
        )
    
    #p.wait()

if __name__ == "__main__":
    fire.Fire()