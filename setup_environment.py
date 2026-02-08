#!/usr/bin/env python3
"""
Environment setup script for Neo4j LangGraph chain
This script checks dependencies and pulls required Ollama models
"""

import subprocess
import sys
import time
import os
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_and_install_packages():
    """Check and install required Python packages"""
    required_packages = [
        'langgraph',
        'ollama', 
        'neo4j',
        'pydantic'
    ]
    
    print("\n📦 Checking Python packages...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def configure_wsl_networking():
    """Configure WSL networking for Ollama access"""
    print("\n🌐 Configuring WSL networking for Ollama...")
    
    try:
        # Get WSL IP address
        result = subprocess.run(['wsl', 'hostname', '-I'], 
                              capture_output=True, text=True, check=True)
        wsl_ips = result.stdout.strip().split()
        wsl_ip = wsl_ips[0]  # Use the first IP
        print(f"WSL IP Address: {wsl_ip}")
        
        # Test connectivity to WSL
        ping_result = subprocess.run(['ping', '-n', '1', wsl_ip], 
                                   capture_output=True, text=True)
        if ping_result.returncode == 0:
            print("✅ Network connectivity to WSL confirmed")
        else:
            print("❌ Cannot ping WSL IP")
            return None
            
        # Test if Ollama is accessible via WSL IP
        try:
            response = requests.get(f"http://{wsl_ip}:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Can access Ollama via WSL IP")
                # Set environment variable for the session
                os.environ['OLLAMA_HOST'] = f"{wsl_ip}:11434"
                return wsl_ip
        except requests.RequestException as e:
            print(f"⚠️  Cannot access Ollama via WSL IP: {e}")
            
        # Try localhost as fallback
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Can access Ollama via localhost")
                os.environ['OLLAMA_HOST'] = "localhost:11434"
                return "localhost"
        except requests.RequestException as e:
            print(f"⚠️  Cannot access Ollama via localhost: {e}")
            
        print("❌ Cannot access Ollama service through any method")
        print("💡 Try restarting Ollama in WSL with: wsl OLLAMA_HOST=0.0.0.0:11434 ollama serve")
        return None
        
    except Exception as e:
        print(f"❌ WSL networking configuration failed: {e}")
        return None

def check_ollama_service():
    """Check if Ollama service is running and accessible"""
    print("\n🦙 Checking Ollama service...")
    
    # Check if OLLAMA_HOST is already set
    ollama_host = os.environ.get('OLLAMA_HOST', 'localhost:11434')
    print(f"Using Ollama host: {ollama_host}")
    
    try:
        import ollama
        # Test basic connectivity
        response = ollama.list()
        print("✅ Ollama service is running and accessible")
        return True
    except Exception as e:
        print(f"❌ Ollama service not accessible: {e}")
        print("Please ensure Ollama is running in WSL with proper network configuration")
        return False

def pull_required_models():
    """Check for required Ollama models (using existing llama2:7b)"""
    required_models = [
        'llama2:7b'    # Using existing model for both chunking and Cypher generation
    ]
    
    print("\n🦙 Checking required Ollama models...")
    
    for model in required_models:
        try:
            print(f"Checking {model}...")
            import ollama
            # Try to get model info
            try:
                model_info = ollama.show(model)
                print(f"✅ {model} is available")
                # Display model size info
                if 'details' in model_info and 'parameter_size' in model_info['details']:
                    print(f"   Size: {model_info['details']['parameter_size']}")
            except Exception as e:
                print(f"❌ {model} not found: {e}")
                print("💡 Please ensure the model is available in your Ollama installation")
                return False
        except Exception as e:
            print(f"❌ Error checking {model}: {e}")
            return False
    
    print("✅ All required models are available")
    return True

def check_docker_and_neo4j():
    """Check Docker and Neo4j setup"""
    print("\n🐳 Checking Docker and Neo4j...")
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker available: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found. Please install Docker first.")
        return False
    
    # Check if Neo4j container is running
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=neo4j-local', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        if 'neo4j-local' in result.stdout:
            print("✅ Neo4j container is running")
            return True
        else:
            print("⚠️  Neo4j container not running")
            print("Starting Neo4j with: docker-compose up -d")
            try:
                subprocess.run(['docker-compose', 'up', '-d'], check=True)
                print("✅ Neo4j started successfully")
                time.sleep(10)  # Wait for Neo4j to initialize
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to start Neo4j container")
                return False
    except subprocess.CalledProcessError:
        print("❌ Error checking Docker containers")
        return False

def test_neo4j_connection():
    """Test connection to Neo4j"""
    print("\n🔗 Testing Neo4j connection...")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message")
            message = result.single()["message"]
            print(f"✅ {message}")
            return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("Make sure Neo4j is running and credentials are correct")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

def main():
    """Main setup function"""
    print("🚀 Setting up Neo4j LangGraph Environment")
    print("=" * 50)
    
    # Configure WSL networking first
    wsl_host = configure_wsl_networking()
    if not wsl_host:
        print("❌ WSL networking configuration failed")
        sys.exit(1)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_and_install_packages),
        ("Ollama Service", check_ollama_service),
        ("Ollama Models", pull_required_models),
        ("Docker & Neo4j", check_docker_and_neo4j),
        ("Neo4j Connection", test_neo4j_connection)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}")
        print("-" * 30)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All checks passed! Environment is ready.")
        print(f"🔧 Ollama is accessible at: {os.environ.get('OLLAMA_HOST', 'unknown')}")
        print("\nNext steps:")
        print("1. Run the main processing script: python neo4j_langgraph_chain.py")
        print("2. Or import and use the process_text_to_neo4j function")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()