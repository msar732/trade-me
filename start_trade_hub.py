#!/usr/bin/env python3
"""
Trade Hub - Startup Script
Simple script to start Trade Hub with dependency checking
"""

import sys
import os
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Try to install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Essential packages that are usually available
    essential_packages = [
        'flask',
        'sqlalchemy',
        'werkzeug'
    ]
    
    for package in essential_packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"⚠️  {package} installation failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⚠️  {package} installation timed out")
        except Exception as e:
            print(f"⚠️  {package} installation error: {e}")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required = {
        'flask': 'Flask web framework',
        'sqlalchemy': 'Database ORM',
        'werkzeug': 'WSGI utilities'
    }
    
    available = {}
    for package, description in required.items():
        try:
            __import__(package)
            available[package] = True
            print(f"✅ {package}: {description}")
        except ImportError:
            available[package] = False
            print(f"❌ {package}: {description} - NOT AVAILABLE")
    
    return available

def create_minimal_app():
    """Create a minimal version of Trade Hub that works without all dependencies"""
    print("🔧 Creating minimal Trade Hub version...")
    
    minimal_app = '''
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from urllib.parse import parse_qs, urlparse

class TradeHubHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade Hub - Your Ultimate Trading Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #f59e0b;
            --success-color: #10b981;
        }
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .hero-section { background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%); padding: 4rem 0; }
        .category-card { transition: transform 0.3s ease; }
        .category-card:hover { transform: translateY(-4px); }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="bg-primary text-white py-3">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <h1 class="h3 mb-0"><i class="bi bi-shop"></i> Trade Hub</h1>
                <div class="d-flex gap-2">
                    <a href="#" class="btn btn-outline-light btn-sm">Login</a>
                    <a href="#" class="btn btn-warning btn-sm">Register</a>
                </div>
            </div>
        </div>
    </header>

    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
        <div class="container">
            <div class="navbar-nav me-auto">
                <a class="nav-link active" href="#"><i class="bi bi-house"></i> Home</a>
                <a class="nav-link" href="#"><i class="bi bi-building"></i> Property</a>
                <a class="nav-link" href="#"><i class="bi bi-car-front"></i> Motors</a>
                <a class="nav-link" href="#"><i class="bi bi-briefcase"></i> Jobs</a>
                <a class="nav-link" href="#"><i class="bi bi-tools"></i> Services</a>
                <a class="nav-link" href="#"><i class="bi bi-cart"></i> Marketplace</a>
            </div>
            <a href="#" class="btn btn-success"><i class="bi bi-plus-circle"></i> Post Free Ad</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container text-center">
            <h1 class="display-4 fw-bold mb-3">Welcome to Trade Hub</h1>
            <p class="lead mb-4">Your ultimate platform for buying, selling, and discovering amazing deals across India</p>
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="input-group input-group-lg">
                        <input type="text" class="form-control" placeholder="What are you looking for?">
                        <button class="btn btn-warning" type="button">
                            <i class="bi bi-search"></i> Find Deals
                        </button>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-md-4">
                    <div class="text-center">
                        <h3 class="text-primary">1000+</h3>
                        <small class="text-muted">ACTIVE USERS</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="text-center">
                        <h3 class="text-primary">5000+</h3>
                        <small class="text-muted">LIVE LISTINGS</small>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="text-center">
                        <h3 class="text-primary">8+</h3>
                        <small class="text-muted">CATEGORIES</small>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Categories -->
    <section class="py-5">
        <div class="container">
            <h2 class="text-center mb-5">Explore Categories</h2>
            <div class="row g-4">
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">🏠</div>
                            <h5>Property</h5>
                            <p class="text-muted">Real Estate & Properties</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">🚗</div>
                            <h5>Motors</h5>
                            <p class="text-muted">Vehicles & Automotive</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">💼</div>
                            <h5>Jobs</h5>
                            <p class="text-muted">Employment Opportunities</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">🔧</div>
                            <h5>Services</h5>
                            <p class="text-muted">Professional Services</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">🛒</div>
                            <h5>Marketplace</h5>
                            <p class="text-muted">Buy & Sell Items</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">📱</div>
                            <h5>Electronics</h5>
                            <p class="text-muted">Gadgets & Electronics</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">👕</div>
                            <h5>Fashion</h5>
                            <p class="text-muted">Clothing & Accessories</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 col-sm-6">
                    <div class="card category-card h-100 text-center">
                        <div class="card-body">
                            <div class="display-1 mb-3">🏡</div>
                            <h5>Home & Garden</h5>
                            <p class="text-muted">Home Improvement</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-dark text-white py-4">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5 class="text-warning">Trade Hub</h5>
                    <p>Your ultimate trading platform for buying, selling, and discovering amazing deals across India.</p>
                </div>
                <div class="col-md-6">
                    <h6>Quick Links</h6>
                    <div class="d-flex flex-wrap gap-3">
                        <a href="#" class="text-light text-decoration-none">Post Free Ad</a>
                        <a href="#" class="text-light text-decoration-none">Search</a>
                        <a href="#" class="text-light text-decoration-none">Help</a>
                        <a href="#" class="text-light text-decoration-none">About</a>
                    </div>
                </div>
            </div>
            <hr class="my-3">
            <div class="text-center">
                <p class="mb-0">&copy; 2024 Trade Hub. All rights reserved. Made with ❤️ in India</p>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 Trade Hub loaded successfully!');
            
            // Add some interactivity
            document.querySelectorAll('.category-card').forEach(card => {
                card.addEventListener('click', function() {
                    alert('Category functionality will be available when full dependencies are installed!');
                });
            });
        });
    </script>
</body>
</html>
            """
            
            self.wfile.write(html.encode())
        else:
            super().do_GET()

def start_minimal_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, TradeHubHandler)
    print(f"🌐 Trade Hub (Minimal) running at http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    start_minimal_server()
'''
    
    with open('/workspace/minimal_trade_hub.py', 'w') as f:
        f.write(minimal_app)
    
    print("✅ Minimal Trade Hub created: minimal_trade_hub.py")

def main():
    """Main startup function"""
    print("🚀 Trade Hub - Startup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    deps = check_dependencies()
    
    if all(deps.values()):
        print("\n✅ All dependencies available! Starting full Trade Hub...")
        try:
            # Import and run the full Trade Hub
            import trade_hub
            print("🌐 Trade Hub running at http://localhost:5000")
            trade_hub.app.run(host='0.0.0.0', port=5000, debug=False)
        except Exception as e:
            print(f"❌ Failed to start full Trade Hub: {e}")
            return False
    else:
        print(f"\n⚠️  Missing dependencies: {[k for k, v in deps.items() if not v]}")
        print("🔧 Would you like to:")
        print("1. Try to install dependencies")
        print("2. Run minimal version (no dependencies required)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            install_dependencies()
            # Re-check dependencies
            deps = check_dependencies()
            if all(deps.values()):
                print("\n✅ Dependencies installed! Starting full Trade Hub...")
                try:
                    import trade_hub
                    trade_hub.app.run(host='0.0.0.0', port=5000, debug=False)
                except Exception as e:
                    print(f"❌ Failed to start full Trade Hub: {e}")
                    return False
            else:
                print("\n⚠️  Some dependencies still missing. Starting minimal version...")
                create_minimal_app()
                exec(open('/workspace/minimal_trade_hub.py').read())
        else:
            print("\n🔧 Starting minimal version...")
            create_minimal_app()
            exec(open('/workspace/minimal_trade_hub.py').read())
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)