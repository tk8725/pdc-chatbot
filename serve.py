import http.server
import socketserver
import webbrowser
import threading
import os
import subprocess
import sys
import time

def start_backend():
    """Start the Flask backend server"""
    try:
        # Change to backend directory
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        os.chdir(backend_dir)
        
        print("Starting backend server on http://localhost:5000")
        print("Backend directory:", os.getcwd())
        
        # Check if app.py exists
        if not os.path.exists('app.py'):
            print("ERROR: app.py not found in backend directory!")
            print("Current files:", os.listdir('.'))
            return
        
        # Start Flask server
        subprocess.run([sys.executable, 'app.py'])
    except Exception as e:
        print(f"Error starting backend: {e}")

def start_frontend():
    """Serve the frontend static files"""
    try:
        # Change to frontend directory
        frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
        os.chdir(frontend_dir)
        
        PORT = 8000
        Handler = http.server.SimpleHTTPRequestHandler
        
        print(f"Frontend directory: {os.getcwd()}")
        print(f"Serving frontend at http://localhost:{PORT}")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{PORT}')
        
        # Open browser in separate thread
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start HTTP server
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
            
    except Exception as e:
        print(f"Error starting frontend: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("Parallel Computing Chatbot - Starting Servers")
    print("=" * 60)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Give backend time to start
    time.sleep(3)
    
    # Start frontend (this will block)
    start_frontend()