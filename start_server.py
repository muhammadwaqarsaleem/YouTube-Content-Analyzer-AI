"""
Startup Script for YouTube Video Analysis Application
Starts Backend API, Frontend, and PO Token Provider servers
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')
warnings.filterwarnings('ignore', message='.*Valid config keys have changed.*')

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import sys
import subprocess
from pathlib import Path
import time


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('yt_dlp', 'yt-dlp'),
        ('youtube_transcript_api', 'youtube-transcript-api'),
        ('tensorflow', 'tensorflow'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')
    ]
    
    missing = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print("\nMissing packages detected!")
        print("Install them with: python -m pip install -r requirements.txt")
        print(f"\nMissing: {', '.join(missing)}")
        
        response = input("\nDo you want to install missing packages now? (y/n): ")
        if response.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("\n✓ Dependencies installed successfully")
        else:
            print("\n⚠ Cannot proceed without required dependencies")
            sys.exit(1)
    
    print("\n✓ All dependencies satisfied\n")


def check_model_files():
    """Check if model files exist"""
    print("Checking model files...")
    
    models = {
        'models/violence_detection/violence_detection_model_resnet.h5': 'Violence Detection Model',
        'models/logistic_regression_model.pkl': 'Category Prediction Model'
    }
    
    warnings = []
    
    for model_path, model_name in models.items():
        if Path(model_path).exists():
            print(f"  ✓ {model_name}: {model_path}")
        else:
            print(f"  ⚠ {model_name} not found: {model_path}")
            warnings.append(f"{model_name} will use fallback prediction")
    
    if warnings:
        print("\n⚠️  Some models are missing:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nThe system will still work but with reduced functionality.\n")
    
    print()


def start_server(host='0.0.0.0', port=8000, reload=False):
    """Start the FastAPI server"""
    print("="*70)
    print("YOUTUBE VIDEO ANALYSIS BACKEND SERVER")
    print("="*70)
    print()
    print(f"Starting server on http://{host}:{port}")
    print()
    print("Endpoints:")
    print("  - API Docs:      http://localhost:{port}/docs")
    print("  - Alternative:   http://localhost:{port}/redoc")
    print("  - API Info:      http://localhost:{port}/api/info")
    print()
    print("Press CTRL+C to stop the server")
    print("="*70)
    print()
    
    try:
        import uvicorn
        
        # Import the FastAPI app
        from api.main import app
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)


def start_all_servers():
    """Start frontend, backend, and PO Token servers"""
    import subprocess
    import threading
    
    print("="*70)
    print("STARTING YOUTUBE VIDEO ANALYSIS SYSTEM")
    print("="*70)
    print()
    print("1. Starting Backend API Server (port 8000)...")
    print()
    
    # Start backend in background
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(Path(__file__).parent)
    )
    
    # Wait for backend to start
    import time
    time.sleep(3)
    
    print()
    print("2. Starting Frontend Server (port 3000)...")
    print()
    
    # Start frontend using built-in HTTP server in a thread
    frontend_thread = threading.Thread(target=start_frontend_server, daemon=True)
    frontend_thread.start()
    
    time.sleep(1)
    
    print()
    print("3. Starting Deno PO Token Server (port 4416)...")
    print()
    
    # Start Deno server
    deno_process = subprocess.Popen(
        ["deno", "run", "-A", "src/main.ts"],
        cwd=str(Path(__file__).parent / "bgutil-ytdlp-pot-provider" / "server")
    )
    
    print("\n" + "="*70)
    print("ALL SERVERS RUNNING")
    print("="*70)
    print("\nAccess Points:")
    print("  🌐 Frontend Dashboard: http://localhost:3000/")
    print("  📖 Backend API Docs:   http://localhost:8000/docs")
    print("  🔑 PO Token Server:    http://localhost:4416")
    print("\nPress CTRL+C to stop all servers")
    print("="*70 + "\n")
    
    try:
        # Wait for backend or deno process to end
        backend_process.wait()
        deno_process.wait()
    except KeyboardInterrupt:
        print("\nStopping all servers...")
        backend_process.terminate()
        deno_process.terminate()


def start_frontend_server():
    """Start only the frontend server (built-in)"""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import socketserver
    
    project_dir = Path(__file__).parent
    frontend_dir = project_dir / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return
    
    os.chdir(frontend_dir)
    
    PORT = 3000
    
    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving frontend at http://localhost:{PORT}")
        print(f"Frontend directory: {frontend_dir}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nFrontend server stopped by user")


def start_deno_server():
    """Start only the deno PO token provider server"""
    project_dir = Path(__file__).parent
    deno_dir = project_dir / "bgutil-ytdlp-pot-provider" / "server"
    
    print(f"Starting Deno server in {deno_dir}")
    
    try:
        process = subprocess.Popen(
            ["deno", "run", "-A", "src/main.ts"],
            cwd=str(deno_dir)
        )
        process.wait()
    except KeyboardInterrupt:
        print("\n\n Deno server stopped by user")
        sys.exit(0)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YouTube Video Analysis Application - Starts Backend, Frontend, and PO Token Provider')
    parser.add_argument('--backend-only', action='store_true', help='Start only backend server')
    parser.add_argument('--frontend-only', action='store_true', help='Start only frontend server')
    parser.add_argument('--deno-only', action='store_true', help='Start only deno/PO token provider server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    # Check prerequisites
    check_dependencies()
    check_model_files()
    
    if args.backend_only:
        # Start only backend
        print("\n🚀 Starting BACKEND server only...\n")
        start_server(args.host, args.port, args.reload)
    elif args.frontend_only:
        # Start only frontend
        print("\n🎨 Starting FRONTEND server only...\n")
        start_frontend_server()
    elif args.deno_only:
        # Start only deno server
        print("\n⚙️ Starting DENO/PO Token Provider server only...\n")
        start_deno_server()
    else:
        # Default: start all three servers
        print("\n" + "="*70)
        print("🚀 STARTING ALL SERVERS")
        print("="*70)
        print("\nThis will start:")
        print("  1. ⚙️ Backend API Server (port 8000)")
        print("  2. 🎨 Frontend Web Server (port 3000)")
        print("  3. 🔑 PO Token Provider Server (port 6969)")
        print("\nPress CTRL+C to stop all servers\n")
        start_all_servers()


if __name__ == "__main__":
    main()
