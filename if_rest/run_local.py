import argparse
import os
from pathlib import Path

import uvicorn


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_env_defaults():
    project_root = _project_root()
    os.environ.setdefault("MODELS_DIR", str(project_root / "models"))
    os.environ.setdefault("ROOT_IMAGES_DIR", str(project_root / "misc"))
    os.environ.setdefault("INFERENCE_BACKEND", "onnx")
    os.environ.setdefault("LOG_LEVEL", "INFO")


def _get_args():
    parser = argparse.ArgumentParser(description="Run InsightFace-REST without Docker.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "18080")), help="Bind port.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", "1")),
        help="Number of Uvicorn workers. Default: NUM_WORKERS env or 1.",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument(
        "--skip-prepare-models",
        action="store_true",
        help="Skip automatic model preparation on startup.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        help="Uvicorn log level. Default: LOG_LEVEL env or info.",
    )
    return parser.parse_args()


def main():
    _configure_env_defaults()
    args = _get_args()
    os.environ["LOG_LEVEL"] = args.log_level.upper()

    if not args.skip_prepare_models:
        from if_rest.prepare_models import prepare_models

        prepare_models()

    uvicorn.run(
        "if_rest.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
