# Run Without Docker

This guide explains how to run `InsightFace-REST` directly on a machine where Docker cannot be installed.

## 1. System Dependencies

On Ubuntu/Debian, install runtime libraries first:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libgomp1 libturbojpeg
```

## 2. Create Python Environment

Python `3.10` is recommended.

Example using `conda`:

```bash
conda create -y -n ifr-local python=3.10
conda activate ifr-local
```

Example using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

Use the local runtime requirements file:

```bash
pip install -r requirements-local.txt
```

## 4. Start Service

From repository root:

```bash
python -m if_rest.run_local --host 0.0.0.0 --port 18080
```

This command will:

- set local defaults for model/image directories
- download and prepare models automatically
- start FastAPI with Uvicorn

Open API docs at:

`http://127.0.0.1:18080/docs`

## 5. Optional Environment Variables

- `MODELS_DIR`: model storage directory (default: `<repo>/models`, or `/models` if present)
- `ROOT_IMAGES_DIR`: local image root for non-URL paths (default: `<repo>/misc`, or `/images` if present)
- `INFERENCE_BACKEND`: backend (`onnx` recommended for local CPU)
- `DET_NAME`: detector model name
- `REC_NAME`: recognition model name
- `GA_NAME`: gender/age model (`None` to disable)
- `MASK_DETECTOR`: mask model (`None` to disable)
- `NUM_WORKERS`: worker count for `if_rest.run_local`

## 6. Startup Tips

- If model download from Google Drive is blocked by your network, place ONNX files manually under `MODELS_DIR/onnx/<model_name>/`.
- If startup fails after code updates, clear Python caches:

```bash
find . | grep -E "(__pycache__|\\.pyc$)" | xargs rm -rf
```
