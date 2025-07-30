call cd /d %~dp0

echo Make sure you have uv installed -> https://github.com/astral-sh/uv#installation
@REM create .env
call copy example.env .env

@REM Create a virtual environment
call uv venv --python 3.11

@REM Install torch with cuda
echo Do you have a GPU?
call pause
echo Install torch with cuda cor the appropriate version for your GPU
echo example for version 12.8 run "uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
echo go to https://pytorch.org/get-started/locally/ to find the appropriate version for your GPU
echo "Install before proceedeing. Open a new terminal and run the command."
call pause

@REM Install dependencies
call uv pip install -e .

@REM Install llama.cpp
echo Open -> https://github.com/ggerganov/llama.cpp or https://github.com/ggerganov/llama.cpp/releases
echo Download the latest release and extract it to the root directory
call pause

@REM launch streamlit
call uv run streamlit run app/ui.py
