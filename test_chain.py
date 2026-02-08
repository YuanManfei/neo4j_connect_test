#!/usr/bin/env python3
"""
Test script for the Neo4j LangGraph processing chain
Demonstrates various use cases and edge cases
"""

import logging
from neo4j_langgraph_chain import process_text_to_neo4j, create_processing_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_processing():
    """Test basic text processing without query execution"""
    print("=== Basic Processing Test ===")
    
    text = "John Doe works at Apple Inc. in Cupertino. He collaborates with Jane Smith from Google."
    
    try:
        results = process_text_to_neo4j(text, execute_queries=False)
        
        print(f"✓ Original text length: {results['original_text_length']}")
        print(f"✓ Number of chunks: {results['number_of_chunks']}")
        print(f"✓ Number of queries: {results['number_of_queries']}")
        
        print("\nGenerated chunks:")
        for i, chunk in enumerate(results['chunks'], 1):
            print(f"  {i}. {chunk}")
            
        print("\nGenerated Cypher queries:")
        for i, query in enumerate(results['cypher_queries'], 1):
            print(f"  {i}. {query}")
            
        return True
        
    except Exception as e:
        print(f"✗ Basic processing failed: {e}")
        return False

def test_complex_relationships():
    """Test processing complex relationship text"""
    print("\n=== Complex Relationships Test ===")
    
    text = """
    Dr. Sarah Chen leads the AI Research department at TechCorp. 
    She mentors Alex Rodriguez and Maria Garcia, both senior researchers.
    The team collaborates with Prof. Michael Brown from Stanford University.
    TechCorp partners with Innovation Labs, where David Kim serves as CTO.
    Sarah published papers with her colleagues and maintains industry connections.
    """
    
    try:
        results = process_text_to_neo4j(text, execute_queries=False)
        
        print(f"✓ Processed {len(results['chunks'])} chunks")
        print(f"✓ Generated {len(results['cypher_queries'])} queries")
        
        # Show first and last queries as examples
        if results['cypher_queries']:
            print(f"\nFirst query: {results['cypher_queries'][0]}")
            if len(results['cypher_queries']) > 1:
                print(f"Last query: {results['cypher_queries'][-1]}")
                
        return True
        
    except Exception as e:
        print(f"✗ Complex relationships test failed: {e}")
        return False

def test_empty_input():
    """Test handling of empty input"""
    print("\n=== Empty Input Test ===")
    
    try:
        results = process_text_to_neo4j("", execute_queries=False)
        print("✓ Empty input handled gracefully")
        print(f"Chunks generated: {len(results['chunks'])}")
        print(f"Queries generated: {len(results['cypher_queries'])}")
        return True
    except Exception as e:
        print(f"✗ Empty input test failed: {e}")
        return False

def test_long_text():
    """Test processing of longer text"""
    print("\n=== Long Text Test ===")
    
    # Generate longer sample text
    text = "Company Alpha employs multiple teams. " * 20  # 20 repetitions
    text += "The engineering team works closely with the product team. "
    text += "Marketing collaborates with sales. "
    text += "The CEO manages all departments. "
    
    try:
        results = process_text_to_neo4j(text, execute_queries=False)
        
        print(f"✓ Processed text of length: {len(text)}")
        print(f"✓ Created {len(results['chunks'])} chunks")
        print(f"✓ Generated {len(results['cypher_queries'])} queries")
        
        avg_chunk_size = sum(len(chunk) for chunk in results['chunks']) / len(results['chunks'])
        print(f"✓ Average chunk size: {avg_chunk_size:.1f} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Long text test failed: {e}")
        return False

def demonstrate_chain_structure():
    """Show the internal structure of the processing chain"""
    print("\n=== Chain Structure Demonstration ===")
    
    try:
        chain = create_processing_chain()
        
        print("✓ Chain created successfully")
        print("✓ Nodes in chain:")
        for node in chain.get_graph().nodes:
            print(f"  - {node}")
            
        print("✓ Edges in chain:")
        for edge in chain.get_graph().edges:
            print(f"  - {edge}")
            
        return True
        
    except Exception as e:
        print(f"✗ Chain structure demonstration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Neo4j LangGraph Chain Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Processing", test_basic_processing),
        ("Complex Relationships", test_complex_relationships),
        ("Empty Input", test_empty_input),
        ("Long Text", test_long_text),
        ("Chain Structure", demonstrate_chain_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The chain is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()