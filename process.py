#!/usr/bin/env python3
"""
Process text.txt document through the Neo4j LangGraph chain
"""

import os
from neo4j_langgraph_chain import process_text_to_neo4j

def process_document():
    """Process the text.txt document through the LangGraph chain"""
    
    # Set OLLAMA_HOST environment variable
    os.environ['OLLAMA_HOST'] = 'localhost:11434'
    
    # Read the text file
    try:
        with open('text.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Successfully read text.txt ({len(text)} characters)")
    except FileNotFoundError:
        print("❌ text.txt file not found")
        return
    except Exception as e:
        print(f"❌ Error reading text.txt: {e}")
        return
    
    # Process through LangGraph chain
    print("🔄 Processing document through LangGraph chain...")
    print(f"📝 Document preview: {text[:100]}...")
    
    try:
        results = process_text_to_neo4j(text, execute_queries=True)
        
        # Display results
        print("\n" + "="*50)
        print("📊 PROCESSING RESULTS")
        print("="*50)
        print(f"📄 Original document length: {results['original_text_length']} characters")
        print(f"📦 Number of chunks created: {results['number_of_chunks']}")
        print(f"🔍 Number of Cypher queries generated: {results['number_of_queries']}")
        print(f"✅ Successfully executed queries: {results['execution_stats']['success']}")
        print(f"❌ Failed queries: {results['execution_stats']['errors']}")
        
        # Show chunk information
        print(f"\n📂 Chunks:")
        for i, chunk in enumerate(results['chunks'], 1):
            print(f"   {i}. Length: {len(chunk)} chars - {chunk[:50]}...")
        
        # Show generated queries
        print(f"\n🔍 Generated Cypher Queries:")
        for i, query in enumerate(results['cypher_queries'], 1):
            print(f"   {i}. {query[:100]}...")
            if len(query) > 100:
                print(f"       ... (truncated, total length: {len(query)} chars)")
        
        print(f"\n🎉 Document processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_document()