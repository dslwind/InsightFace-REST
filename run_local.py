#!/usr/bin/env python
"""
Local runner for InsightFace-REST without Docker.
This script loads environment variables and starts the service.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_env_file(env_file='local.env'):
    """Load environment variables from .env file"""
    env_path = project_root / env_file
    if not env_path.exists():
        print(f"Warning: {env_file} not found. Using default environment variables.")
        return
    
    print(f"Loading environment variables from {env_file}...")
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable if not already set
                if key not in os.environ:
                    os.environ[key] = value

def ensure_models_dir():
    """Ensure models directory exists"""
    models_dir = os.getenv('MODELS_DIR', 'models')
    
    # If relative path, make it absolute relative to project root
    if not os.path.isabs(models_dir):
        models_dir = str(project_root / models_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Update MODELS_DIR to absolute path
    os.environ['MODELS_DIR'] = models_dir
    
    print(f"Models directory: {models_dir}")
    return models_dir

def main():
    """Main entry point"""
    # Load environment variables
    load_env_file()
    
    # Ensure models directory exists
    ensure_models_dir()
    
    # Import after setting up environment
    import subprocess
    import platform
    
    # Detect operating system
    is_windows = platform.system() == 'Windows'
    
    # Get configuration
    num_workers = os.getenv('NUM_WORKERS', '4')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    port = os.getenv('PORT', '18080')
    
    print("\n" + "="*60)
    print("InsightFace-REST Local Runner")
    print("="*60)
    if is_windows:
        print(f"Platform: Windows (using uvicorn)")
        print(f"Note: Multiple workers not supported on Windows with uvicorn")
    else:
        print(f"Workers: {num_workers}")
    print(f"Log Level: {log_level}")
    print(f"Port: {port}")
    print("="*60 + "\n")
    
    # Prepare models first
    print("Preparing models...")
    try:
        from if_rest.prepare_models import prepare_models
        root_dir = os.getenv('MODELS_DIR', 'models')
        prepare_models(root_dir=root_dir)
        print("Models prepared successfully!\n")
    except Exception as e:
        print(f"Error preparing models: {e}")
        print("Continuing anyway...\n")
    
    # Start the server
    print(f"Starting InsightFace-REST on port {port}...")
    print(f"API Documentation: http://localhost:{port}/docs\n")
    
    if is_windows:
        # On Windows, use uvicorn directly (gunicorn doesn't work on Windows)
        cmd = [
            sys.executable, '-m', 'uvicorn',
            'if_rest.api.main:app',
            '--host', '0.0.0.0',
            '--port', port,
            '--log-level', log_level.lower()
        ]
    else:
        # On Linux/Mac, use gunicorn with uvicorn workers
        cmd = [
            sys.executable, '-m', 'gunicorn',
            '--log-level', log_level,
            '-w', num_workers,
            '-k', 'uvicorn.workers.UvicornWorker',
            '--keep-alive', '60',
            '--timeout', '60',
            'if_rest.api.main:app',
            '-b', f'0.0.0.0:{port}'
        ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
