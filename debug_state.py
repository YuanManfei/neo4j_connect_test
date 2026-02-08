#!/usr/bin/env python3
"""
Debug script to understand the state management issue
"""

import logging
from neo4j_langgraph_chain import process_text_to_neo4j, create_processing_chain

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def debug_state_issue():
    """Debug the state management issue"""
    print("🔍 Debugging State Management Issue")
    print("=" * 40)
    
    # Test with minimal example
    text = "Test text"
    
    try:
        print("Creating processing chain...")
        chain = create_processing_chain()
        print("Chain created successfully")
        
        print("\nTesting chain invocation...")
        from neo4j_langgraph_chain import ProcessingState
        initial_state = ProcessingState(original_text=text)
        print(f"Initial state type: {type(initial_state)}")
        print(f"Initial state: {initial_state}")
        
        config = {"configurable": {"thread_id": "debug-session"}}
        
        print("\nInvoking chain...")
        result = chain.invoke(initial_state, config)
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        if hasattr(result, 'chunks'):
            print(f"Chunks: {result.chunks}")
        elif isinstance(result, dict) and 'chunks' in result:
            print(f"Chunks from dict: {result['chunks']}")
        else:
            print("❌ No chunks found in result")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_state_issue()