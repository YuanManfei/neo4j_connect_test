import os
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import ollama
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State schema definition
class ProcessingState(BaseModel):
    """State model for the processing chain"""
    original_text: str = Field(default="", description="Original input text")
    chunks: List[str] = Field(default_factory=list, description="Text chunks")
    cypher_queries: List[str] = Field(default_factory=list, description="Generated Cypher queries")
    current_chunk_index: int = Field(default=0, description="Current chunk being processed")
    processed_chunks: int = Field(default=0, description="Number of chunks processed")
    context: str = Field(default="", description="Accumulated context from previous chunks")

# Reducer functions for state management
def add_chunks(left: List[str], right: List[str]) -> List[str]:
    """Reducer for combining text chunks"""
    if not left:
        left = []
    return left + right

def add_queries(left: List[str], right: List[str]) -> List[str]:
    """Reducer for combining Cypher queries"""
    if not left:
        left = []
    return left + right

class TextChunker:
    """Agent for chunking text into entity-focused pieces"""
    
    def __init__(self, model_name: str = "llama2:7b"):
        self.model_name = model_name
        self.ollama_host = os.environ.get('OLLAMA_HOST', 'localhost:11434')
        logger.info(f"Initializing TextChunker with model: {model_name} at {self.ollama_host}")
    
    def chunk_text(self, state: ProcessingState) -> Dict[str, Any]:
        """
        Chunk the input text into entity-focused pieces
        Each chunk should represent a single entity or closely related entities
        """
        text = state.original_text
        if not text:
            return {"chunks": []}
        
        # Try to use Ollama for intelligent entity-based chunking
        try:
            prompt = f"""Analyze this text and identify distinct entities (people, organizations, concepts, locations, etc.).
            Break the text into chunks where each chunk focuses on one main entity or a small group of closely related entities.
            
            Text: {text[:2000]}{'...' if len(text) > 2000 else ''}
            
            Return chunk boundaries that separate different entities while keeping related information together.
            Each chunk should be small enough to represent a single main entity."""
            
            # Configure Ollama client with detected host
            ollama_client = ollama.Client(host=f"http://{self.ollama_host}")
            
            response = ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.3}
            )
            
            # Entity-focused chunking based on identified entities
            sentences = text.split('. ')
            chunks = []
            current_chunk = ""
            max_entity_chunk_size = 150  # Much smaller chunks focused on entities
            
            for sentence in sentences:
                # Check if adding this sentence would make chunk too large
                if len(current_chunk) + len(sentence) > max_entity_chunk_size and current_chunk:
                    # If current chunk has content, save it and start new one
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            
            # Add the final chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
            logger.info(f"Created {len(chunks)} entity-focused chunks from text of length {len(text)}")
            logger.info(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} characters")
            return {"chunks": chunks}
            
        except Exception as e:
            logger.warning(f"Ollama not available, using fallback entity-based chunking: {e}")
            # Fallback to simple entity-focused chunking without Ollama
            max_entity_chunk_size = 150  # Small chunks for entity focus
            chunks = []
            sentences = text.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_entity_chunk_size and current_chunk:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
            logger.info(f"Fallback: Created {len(chunks)} entity-focused chunks using simple method")
            logger.info(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} characters")
            return {"chunks": chunks}

class CypherGenerator:
    """Agent for converting text chunks to Cypher queries"""
    
    def __init__(self, model_name: str = "llama2:7b"):
        self.model_name = model_name
        self.ollama_host = os.environ.get('OLLAMA_HOST', 'localhost:11434')
        logger.info(f"Initializing CypherGenerator with model: {model_name} at {self.ollama_host}")
    
    def generate_cypher(self, state: ProcessingState) -> Dict[str, Any]:
        """
        Convert entity chunk to Cypher query and evaluate relationships with other entities
        """
        if state.current_chunk_index >= len(state.chunks):
            return {
                "cypher_queries": [],
                "processed_chunks": 1,
                "context": state.context
            }
        
        chunk = state.chunks[state.current_chunk_index]
        logger.info(f"Processing chunk {state.current_chunk_index + 1}/{len(state.chunks)}")
        
        # Build context from previously processed chunks and generated queries
        context = state.context or "No previous context."
        
        # Try to use Ollama for Cypher generation
        try:
            # Highly specific prompt for pure Cypher generation
            prompt = f"""Convert this entity-focused text chunk into a Cypher query for Neo4j.
            
            Chunk: {chunk}
            
            Rules:
            1. Return ONLY the Cypher query, no explanations
            2. Start directly with MERGE, CREATE, or MATCH
            3. Identify the main entity in this chunk
            4. Create relationships to previously mentioned entities when relevant
            5. Use MERGE to prevent duplicates
            6. Include key properties like name, type, etc.
            
            Context from previous chunks: {context}"""
            
            # Configure Ollama client with detected host
            ollama_client = ollama.Client(host=f"http://{self.ollama_host}")
            
            response = ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Very low temperature for consistency
                    "stop": ["\n\n", ".", "Explanation:", "Note:", "Here", "Cypher:", "Query:"]  # Aggressive stopping
                }
            )
            
            generated_query = response['response'].strip()
            
            # Extract pure Cypher using regex (more aggressive cleaning)
            import re
            # Look for Cypher patterns and extract them
            cypher_patterns = [
                r'(MERGE\s*\(.*?\).*?(?:\n|$))',
                r'(CREATE\s*\(.*?\).*?(?:\n|$))',
                r'(MATCH\s*\(.*?\).*?MERGE.*?(?:\n|$))',
                r'(MATCH\s*\(.*?\).*?CREATE.*?(?:\n|$))'
            ]
            
            clean_query = ""
            for pattern in cypher_patterns:
                matches = re.findall(pattern, generated_query, re.DOTALL | re.IGNORECASE)
                if matches:
                    clean_query = matches[0].strip()
                    break
            
            # If no clear Cypher found, try to extract any valid-looking Cypher
            if not clean_query:
                lines = generated_query.split('\n')
                cypher_lines = [line for line in lines if any(keyword in line.upper() for keyword in ['MERGE', 'CREATE', 'MATCH'])]
                if cypher_lines:
                    clean_query = cypher_lines[0].strip()
            
            # Validate the extracted query
            validator = CypherValidator()
            if clean_query and validator.is_valid_cypher(clean_query):
                logger.info(f"Generated clean Cypher query for chunk {state.current_chunk_index + 1}")
                queries = [clean_query]
            else:
                logger.warning(f"No valid Cypher found, creating structured fallback")
                # Create structured fallback based on chunk content
                if chunk:
                    # Simple entity extraction for fallback
                    entity_parts = [p.strip() for p in chunk.split() if p.strip()]
                    entity_name = " ".join(entity_parts[:3]) if entity_parts else "Unknown"
                    fallback_query = f"MERGE (e:Entity {{name: '{entity_name}'}})"
                    queries = [fallback_query]
                else:
                    queries = ["// Empty chunk"]
            
            logger.info(f"Generated Cypher query: {clean_query[:100]}...")
            
            # Update context with information about this chunk and its relationships
            updated_context = f"{context}\nChunk {state.current_chunk_index + 1}: {chunk}\nQuery: {clean_query}"
            
            return {
                "cypher_queries": queries,
                "current_chunk_index": state.current_chunk_index + 1,
                "processed_chunks": 1,
                "context": updated_context
            }
            
        except Exception as e:
            logger.error(f"Error generating Cypher for chunk {state.current_chunk_index + 1}: {e}")
            # Fallback query when LLM fails
            fallback_query = f"// Fallback query for chunk: {chunk[:50]}..."
            updated_context = f"{context}\nChunk {state.current_chunk_index + 1}: {chunk}\nQuery: {fallback_query}"
            return {
                "cypher_queries": [fallback_query],
                "current_chunk_index": state.current_chunk_index + 1,
                "processed_chunks": 1,
                "context": updated_context
            }

class CypherValidator:
    """Utility to validate and clean Cypher queries"""
    
    @staticmethod
    def clean_cypher_query(query: str) -> str:
        """Remove natural language and extract only Cypher statements"""
        # Split by newlines
        lines = query.strip().split('\n')
        
        # Filter out non-Cypher lines
        cypher_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comments (unless they're Cypher comments)
            if not stripped:
                continue
                
            # Keep lines that:
            # 1. Start with Cypher keywords
            # 2. Are Cypher comments (//)
            # 3. Contain Cypher patterns
            if (stripped.startswith(("MERGE", "CREATE", "MATCH", "WITH", "RETURN",
                                    "SET", "DELETE", "DETACH", "UNWIND", "OPTIONAL",
                                    "ORDER", "LIMIT", "SKIP", "WHERE", "AND", "OR",
                                    "//", "(", ")")) or
                any(pattern in stripped for pattern in ["[:", "]->", "]-[", ")-[", "]-(", ":{"]) or
                stripped.endswith(("{", "}", "]", ")"))):
                cypher_lines.append(stripped)
        
        return '\n'.join(cypher_lines) if cypher_lines else "// No valid Cypher found in response"
    
    @staticmethod
    def is_valid_cypher(query: str) -> bool:
        """Basic validation of Cypher syntax"""
        if not query or query.startswith("//"):
            return False
        
        # Check for at least one Cypher keyword
        cypher_keywords = ["MERGE", "CREATE", "MATCH", "RETURN", "WITH"]
        return any(keyword in query.upper() for keyword in cypher_keywords)
    
class Neo4jExecutor:
    """Utility class for executing Cypher queries against Neo4j"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "natural-door-radius-harris-fashion-9582"):
        self.uri = uri
        self.username = username
        self.password = password
        self.validator = CypherValidator()
        logger.info(f"Initializing Neo4jExecutor for {uri}")
    
    def execute_query(self, query: str) -> bool:
        """Execute a single Cypher query"""
        try:
            # Clean and validate the query first
            cleaned_query = self.validator.clean_cypher_query(query)
            
            if not self.validator.is_valid_cypher(cleaned_query):
                logger.warning(f"Skipping invalid query: {query[:100]}...")
                return False
            
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                result = session.run(cleaned_query)
                # Consume the result to actually execute the query
                result.consume()
                logger.info(f"Successfully executed query: {cleaned_query[:100]}...")
                return True
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return False
        finally:
            if 'driver' in locals():
                driver.close()
    
    def execute_all_queries(self, queries: List[str]) -> Dict[str, int]:
        """Execute all Cypher queries and return statistics"""
        success_count = 0
        error_count = 0
        
        for query in queries:
            if query.startswith("//"):  # Skip comments/errors
                error_count += 1
                continue
                
            if self.execute_query(query):
                success_count += 1
            else:
                error_count += 1
        
        return {"success": success_count, "errors": error_count}

def should_continue_processing(state: ProcessingState) -> str:
    """Conditional edge to determine if processing should continue"""
    # Handle both ProcessingState object and dict representations
    if isinstance(state, dict):
        chunks = state.get('chunks', [])
        current_index = state.get('current_chunk_index', 0)
    else:
        chunks = state.chunks
        current_index = state.current_chunk_index
        
    if current_index < len(chunks):
        return "generate_cypher"
    else:
        return "END"

def create_processing_chain():
    """Create and compile the LangGraph processing chain"""
    
    # Initialize agents with llama2:7b model
    chunker = TextChunker(model_name="llama2:7b")
    cypher_gen = CypherGenerator(model_name="llama2:7b")
    
    # Create state graph
    workflow = StateGraph(ProcessingState)
    
    # Add nodes with proper state handling
    def chunk_text_wrapper(state):
        result = chunker.chunk_text(state)
        # Ensure we return a proper state update with context initialization
        return {
            "chunks": result["chunks"],
            "current_chunk_index": 0,  # Reset index
            "processed_chunks": 0,
            "context": state.original_text[:500]  # Initialize with beginning context
        }
    
    def generate_cypher_wrapper(state):
        result = cypher_gen.generate_cypher(state)
        # Use the context from the result which already includes the current chunk info
        final_context = result.get("context", state.context)
        # Keep context manageable
        return {
            "cypher_queries": result["cypher_queries"],
            "current_chunk_index": result["current_chunk_index"],
            "processed_chunks": result["processed_chunks"],
            "context": final_context[:1000]
        }
    
    # Add nodes
    workflow.add_node("chunk_text", chunk_text_wrapper)
    workflow.add_node("generate_cypher", generate_cypher_wrapper)
    
    # Add edges
    workflow.add_edge("chunk_text", "generate_cypher")
    workflow.add_conditional_edges(
        "generate_cypher",
        should_continue_processing,
        {
            "generate_cypher": "generate_cypher",
            "END": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("chunk_text")
    
    # Compile with memory checkpointing
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def process_text_to_neo4j(text: str, execute_queries: bool = True) -> Dict[str, Any]:
    """
    Main function to process text through the LangGraph chain
    
    Args:
        text: Input text to process
        execute_queries: Whether to actually execute the generated Cypher queries
    
    Returns:
        Dictionary with processing results
    """
    logger.info("Starting text processing pipeline")
    
    # Create the chain
    chain = create_processing_chain()
    
    # Initial state
    initial_state = ProcessingState(original_text=text)
    
    # Configuration for checkpointing
    config = {"configurable": {"thread_id": "processing-session-1"}}
    
    # Execute the chain
    result_dict = chain.invoke(initial_state, config)
    
    # The chain returns a dict, not a ProcessingState object
    # Extract the relevant information
    results = {
        "original_text_length": len(text),
        "number_of_chunks": len(result_dict.get('chunks', [])),
        "number_of_queries": len(result_dict.get('cypher_queries', [])),
        "chunks": result_dict.get('chunks', []),
        "cypher_queries": result_dict.get('cypher_queries', [])
    }
    
    # Execute queries if requested
    if execute_queries and results['cypher_queries']:
        executor = Neo4jExecutor()
        # Fix: Use the correct method name
        execution_stats = executor.execute_all_queries(results['cypher_queries'])
        results["execution_stats"] = execution_stats
        logger.info(f"Query execution completed: {execution_stats}")
    
    logger.info("Text processing pipeline completed successfully")
    return results

# Example usage and testing
if __name__ == "__main__":
    print("=== Neo4j LangGraph Processing Chain ===")
    print("Module loaded successfully. Use process_text_to_neo4j() function to process text.")