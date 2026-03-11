"""
Frontend Development Server
Serves static HTML/CSS/JS files on port 3000
"""

import http.server
import socketserver
from pathlib import Path
import sys

PORT = 3000
DIRECTORY = "frontend"


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[Frontend] {args[0]}")


def main():
    # Change to script directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("="*70)
        print("YOUTUBE VIDEO ANALYSIS - FRONTEND SERVER")
        print("="*70)
        print(f"\n Frontend serving at: http://localhost:{PORT}/")
        print(f" Backend API at:      http://localhost:8000/")
        print(f" API Documentation:   http://localhost:8000/docs")
        print("\n Press CTRL+C to stop the server")
        print("="*70 + "\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n Frontend server stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    main()
