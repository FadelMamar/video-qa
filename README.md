<div align="center">
  <img src="image.png" alt="Surveillance Video Analysis PoC Banner" width="600"/>
</div>

# Surveillance Video Analysis PoC

Leverage AI-powered insights to analyze surveillance videos with advanced activity detection, object recognition, and video summarization. This project provides both a command-line interface (CLI) and a modern Streamlit web UI for interactive analysis.

---

## Features

- **Video Analysis**: Extracts key activities, objects, and summaries from surveillance videos.
- **AI Model Integration**: Supports Vision-Language Models (VLM), LLMs, and CLIP for deep video understanding.
- **Object Detection**: Uses YOLO-based models for frame-level annotation.
- **Streamlit UI**: User-friendly web interface for uploading, analyzing, and visualizing results.
- **CLI Tools**: Command-line utilities for batch processing and automation.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/video-qa.git
cd video-qa
```

### 2. Install [uv](https://github.com/astral-sh/uv)

#### On Linux/macOS:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

#### On Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

For more details or troubleshooting, see the [uv installation guide](https://github.com/astral-sh/uv#installation).

### 3. Install Dependencies (using uv)

It's recommended to use a virtual environment.

```bash
uv venv .venv
uv sync
uv pip install -e .
```

### 3.1. Download and Set Up llama.cpp (for Llama server support)

If you plan to use Llama-based models, you need the [llama.cpp](https://github.com/ggerganov/llama.cpp) server executable.

#### Option 1: Download Pre-built Executable

- Visit the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases) and download the latest pre-built binary for your platform:
  - **Windows:** Download the `.zip` file for Windows, extract it, and locate `llama-server.exe`.
  - **Linux/macOS:** Download the appropriate archive for your OS, extract it, and locate `llama-server`.

After downloading, set the `LLAMA_SERVER_PATH` variable in your `.env` to the path of the `llama-server` (or `llama-server.exe` on Windows) executable, e.g.:
```
LLAMA_SERVER_PATH=/path/to/llama-server
```

#### Option 2: Build from Source

- See instructions above or in the [llama.cpp documentation](https://github.com/ggerganov/llama.cpp#build).

### 4. Environment Variables

This project uses environment variables for configuration. An example configuration file is provided as `example.env` in the project root.

#### Steps:
1. Copy the example file to create your own `.env`:
   ```bash
   cp example.env .env
   ```
   On Windows (PowerShell):
   ```powershell
   copy example.env .env
   ```
2. Open `.env` in a text editor and update the values as needed for your environment.

#### Key Environment Variables

Below are some important environment variables you may want to configure in your `.env` file:

- `HF_TOKEN` — HuggingFace API token (if required)
- `OPENAI_API_KEY` — OpenAI API key (default: `sk-no-key-required`)
- `HOST` — Host address for local endpoints (default: `localhost`)
- `VLM_PORT` — Port for the Vision-Language Model endpoint (e.g., `8000`)
- `LLM_PORT` — Port for the Language Model endpoint (e.g., `8008`)
- `CTX_SIZE` — Context size for model inference (e.g., `20000`)
- `MODEL_NAME` — Vision-Language Model identifier or path (e.g., `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m`)
- `LLM_MODEL` — Language Model identifier or path (e.g., `ggml-org/Qwen3-0.6B-GGUF:f16`)
- `LLAMA_SERVER_PATH` — Path to the Llama server executable (if used)
- `LLAMA_SERVER_LOG` — Path to the Llama server log file (e.g., `llama_server.log`)
- `TEMPERATURE` — Sampling temperature for model inference (e.g., `0.7`)
- `DSPY_CACHEDIR` — Directory for DSPy cache (e.g., `.cache_dspy`)
- `VIDEO_PREPROCESSED_DIR` — Directory for preprocessed videos (e.g., `preprocessed_videos`)
- `CLIP_MODEL` — Path or identifier for your CLIP model (if required)

Example `.env` values:

Refer to `example.env` for all available options and their default/example values.

---

## Usage

### 1. Launch the Streamlit UI

```bash
streamlit run app/ui.py
```

- Open your browser at the provided URL (usually http://localhost:8501).
- Upload a video or provide a path to a local video file.
- Configure analysis options in the sidebar.
- Click "Start Analysis" to view results and logs.

### 2. Command-Line Interface

Analyze a video directly from the terminal:

```bash
python cli.py analyze <path_to_video> --args='{"sample_freq":5, ...}' --activity_analysis=True
```

- Replace `<path_to_video>` with your video file.
- Adjust `--args` as needed (see `PredictionConfig` in `watcher/config.py`).

#### Launch AI Model Endpoints

```bash
python cli.py launch_vlm --model_name="your-vlm-model" --port=8001 --ctx_size=20000
```

---

## Project Structure

```
video-qa/
  app/                # Streamlit UI
  src/watcher/        # Core analysis, detection, and utils
  cli.py              # Command-line interface
  models/        # Model checkpoints
  preprocessed_videos/
  tests/              # Test scripts
  README.md
  pyproject.toml
```

---

## Requirements

- Python 3.11
- Streamlit
- torch, torchvision
- ultralytics
- dotenv
- Other dependencies as listed in `pyproject.toml`

---

## Example

1. **Upload or specify a video in the UI.**
2. **Click "Start Analysis".**
3. **View key insights, frame-level metadata, and logs.**

---

## Customization

- **Models**: Place your YOLO or VLM models in the specified paths and update `.env`.
- **Sampling Frequency**: Adjust in the UI or CLI for faster or more detailed analysis.
- **Device**: By default, uses CPU. For GPU, ensure torch detects CUDA.

---

## Troubleshooting

- If you see errors about missing models, check your `.env` paths.
- For large videos, use the "Video path" input instead of uploading.
- Logs are available in the UI under "Logs" expanders.

---

## Contributing

Pull requests and issues are welcome! Please open an issue for bugs or feature requests.

---

## License

[MIT License](LICENSE)

