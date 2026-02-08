#!/usr/bin/env python3
"""
Comprehensive connection testing script for Neo4j LangGraph chain
Tests Ollama, Neo4j, and overall system connectivity
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def test_python_environment():
    """Test Python environment and dependencies"""
    print("🔍 Testing Python Environment")
    print("=" * 40)
    
    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    
    # Check required packages
    required_packages = ['langgraph', 'ollama', 'neo4j', 'pydantic']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def test_wsl_connectivity():
    """Test WSL network connectivity"""
    print("\n🌐 Testing WSL Connectivity")
    print("=" * 40)
    
    try:
        # Get WSL IP
        result = subprocess.run(['wsl', 'hostname', '-I'], 
                              capture_output=True, text=True, check=True)
        wsl_ips = result.stdout.strip().split()
        wsl_ip = wsl_ips[0]
        print(f"WSL IP: {wsl_ip}")
        
        # Test ping
        ping_result = subprocess.run(['ping', '-n', '1', wsl_ip], 
                                   capture_output=True, text=True)
        if ping_result.returncode == 0:
            print("✅ Network connectivity to WSL OK")
        else:
            print("❌ Cannot ping WSL")
            return False, None
            
        return True, wsl_ip
        
    except Exception as e:
        print(f"❌ WSL connectivity test failed: {e}")
        return False, None

def test_ollama_connection(wsl_ip):
    """Test Ollama connection"""
    print("\n🦙 Testing Ollama Connection")
    print("=" * 40)
    
    # Test different connection methods
    hosts_to_test = [
        f"{wsl_ip}:11434",
        "localhost:11434",
        "127.0.0.1:11434"
    ]
    
    working_host = None
    
    for host in hosts_to_test:
        try:
            print(f"Testing {host}...")
            response = requests.get(f"http://{host}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama accessible at {host}")
                working_host = host
                # Set environment variable for the session
                os.environ['OLLAMA_HOST'] = host
                break
            else:
                print(f"❌ {host} returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ {host} connection failed: {e}")
    
    if not working_host:
        print("❌ No working Ollama connection found")
        print("💡 Try: wsl OLLAMA_HOST=0.0.0.0:11434 ollama serve")
        return False
    
    # Test model availability
    try:
        import ollama
        models = ollama.list()
        print("Available models:")
        for model in models['models']:
            print(f"  - {model['name']} ({model['size']})")
        
        # Check if required model exists
        required_model = "llama2:7b"
        model_names = [m['name'] for m in models['models']]
        if required_model in model_names:
            print(f"✅ Required model {required_model} available")
        else:
            print(f"❌ Required model {required_model} not found")
            return False
            
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False
    
    return True

def test_neo4j_connection():
    """Test Neo4j connection with provided credentials"""
    print("\n🔗 Testing Neo4j Connection")
    print("=" * 40)
    
    try:
        from neo4j import GraphDatabase
        
        # Test connection
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", "natural-door-radius-harris-fashion-9582")
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message")
            message = result.single()["message"]
            print(f"✅ {message}")
            
            # Test basic query
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            count = result.single()["node_count"]
            print(f"✅ Database contains {count} nodes")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_docker_compose():
    """Test Docker Compose setup"""
    print("\n🐳 Testing Docker Compose")
    print("=" * 40)
    
    try:
        # Check if docker-compose.yml exists
        if not Path('docker-compose.yml').exists():
            print("❌ docker-compose.yml not found")
            return False
            
        print("✅ docker-compose.yml found")
        
        # Check if Neo4j container is running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=neo4j-local', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        
        if 'neo4j-local' in result.stdout:
            print("✅ Neo4j container is running")
            return True
        else:
            print("⚠️  Neo4j container not running")
            print("💡 Start with: docker-compose up -d")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker command failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found")
        return False

def test_langgraph_chain():
    """Test the complete LangGraph chain"""
    print("\n⛓️  Testing LangGraph Chain")
    print("=" * 40)
    
    try:
        from neo4j_langgraph_chain import process_text_to_neo4j
        
        # Test with simple text
        test_text = "Alice knows Bob. Bob works at Google."
        print(f"Processing: {test_text}")
        
        results = process_text_to_neo4j(test_text, execute_queries=False)
        
        print(f"✅ Processing successful")
        print(f"   Chunks: {len(results['chunks'])}")
        print(f"   Queries: {len(results['cypher_queries'])}")
        
        if results['cypher_queries']:
            print(f"   Sample query: {results['cypher_queries'][0][:100]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ LangGraph chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all connection tests"""
    print("🚀 Comprehensive Connection Testing")
    print("=" * 50)
    
    all_tests_passed = True
    test_results = []
    
    # Run tests in order
    tests = [
        ("Python Environment", test_python_environment),
        ("WSL Connectivity", lambda: test_wsl_connectivity()[0]),
        ("Docker Compose", test_docker_compose),
        ("Neo4j Connection", test_neo4j_connection),
        ("Ollama Connection", lambda: test_ollama_connection(test_wsl_connectivity()[1])),
        ("LangGraph Chain", test_langgraph_chain)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            test_results.append((test_name, result))
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 CONNECTION TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    if all_tests_passed:
        print("\n🎉 All connection tests passed!")
        print("✅ Your Neo4j LangGraph chain is ready to use!")
        print("\nNext steps:")
        print("1. Run: python neo4j_langgraph_chain.py")
        print("2. Or use the process_text_to_neo4j() function in your code")
    else:
        print("\n⚠️  Some connection tests failed.")
        print("Please address the failed tests above.")
        
        # Provide specific troubleshooting advice
        failed_tests = [name for name, result in test_results if not result]
        if "Ollama Connection" in failed_tests:
            print("\n💡 Ollama troubleshooting:")
            print("   - Ensure Ollama is running in WSL: wsl ollama serve")
            print("   - Check if llama2:7b model is available: wsl ollama list")
            print("   - Configure external access: wsl OLLAMA_HOST=0.0.0.0:11434 ollama serve")
        
        if "Neo4j Connection" in failed_tests:
            print("\n💡 Neo4j troubleshooting:")
            print("   - Start Neo4j: docker-compose up -d")
            print("   - Check container status: docker ps")
            print("   - Verify credentials in neo4j_langgraph_chain.py")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)