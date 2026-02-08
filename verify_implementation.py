#!/usr/bin/env python3
"""
Simple verification script for the Neo4j LangGraph implementation
"""

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from neo4j_langgraph_chain import (
            process_text_to_neo4j, 
            create_processing_chain,
            TextChunker,
            CypherGenerator,
            ProcessingState
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_chain_creation():
    """Test that the processing chain can be created"""
    try:
        from neo4j_langgraph_chain import create_processing_chain
        chain = create_processing_chain()
        print("✅ Chain creation successful")
        return True
    except Exception as e:
        print(f"❌ Chain creation failed: {e}")
        return False

def test_simple_processing():
    """Test simple text processing"""
    try:
        from neo4j_langgraph_chain import process_text_to_neo4j
        text = "Alice knows Bob. Bob works at Google."
        results = process_text_to_neo4j(text, execute_queries=False)
        
        print("✅ Simple processing successful")
        print(f"   Text length: {results['original_text_length']}")
        print(f"   Chunks: {len(results['chunks'])}")
        print(f"   Queries: {len(results['cypher_queries'])}")
        return True
    except Exception as e:
        print(f"❌ Simple processing failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🔍 Verifying Neo4j LangGraph Implementation")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Chain Creation", test_chain_creation),
        ("Simple Processing", test_simple_processing)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 VERIFICATION RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Implementation verified successfully!")
        print("\nReady to use:")
        print("- Run 'python setup_environment.py' to prepare environment")
        print("- Run 'python test_chain.py' for comprehensive tests")
        print("- Use process_text_to_neo4j() function for processing")
    else:
        print("⚠️  Some verification tests failed")

if __name__ == "__main__":
    main()