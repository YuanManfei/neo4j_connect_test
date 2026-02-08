# Neo4j LangGraph Processing Chain

This project implements a LangGraph-based processing chain that chunks text and converts it to Cypher queries for insertion into Neo4j. It uses Ollama for local LLM inference and Docker for Neo4j deployment.

## 🏗️ Architecture

The processing chain consists of three main components:

1. **TextChunker Agent** - Uses `llama2:7b` model to intelligently chunk input text
2. **CypherGenerator Agent** - Uses `llama2:7b` model to convert chunks to Cypher queries  
3. **Neo4jExecutor** - Executes generated Cypher queries against local Neo4j instance

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Ollama service running locally with `llama2:7b` model
- Virtual environment named `kg_env`

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate your virtual environment
conda activate kg_env  # or source kg_env/bin/activate

# Run the setup script
python setup_environment.py
```

This script will:
- Check Python version and dependencies
- Install required packages
- Verify Ollama service availability with `llama2:7b` model
- Configure WSL networking if needed
- Start Neo4j via Docker Compose
- Test Neo4j connection

### 2. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure llama2:7b model is available
ollama list  # Should show llama2:7b

# Start Neo4j
docker-compose up -d

# Wait for Neo4j to initialize (about 30 seconds)
```

### 3. Run the Processing Chain

```python
from neo4j_langgraph_chain import process_text_to_neo4j

# Sample text
text = """
Alice Johnson works at Google and lives in San Francisco. 
She knows Bob Smith who works at Microsoft in Seattle.
Bob is friends with Charlie Brown who studies at Stanford University.
"""

# Process without executing queries (for testing)
results = process_text_to_neo4j(text, execute_queries=False)
print(f"Generated {len(results['cypher_queries'])} Cypher queries")

# Process and execute queries
results = process_text_to_neo4j(text, execute_queries=True)
print(f"Execution stats: {results['execution_stats']}")
```

## 🔧 Configuration

### Model Selection

Based on your requirement, we're using the existing `llama2:7b` model for both agents:

- **Both Chunking and Cypher Generation**: `llama2:7b`

You can modify this in the `TextChunker` and `CypherGenerator` classes:

```python
chunker = TextChunker(model_name="llama2:7b")  
cypher_gen = CypherGenerator(model_name="llama2:7b")
```

### Neo4j Configuration

Default connection settings:
- URI: `bolt://localhost:7687`
- Username: `neo4j`
- Password: `natural-door-radius-harris-fashion-9582`

Modify in `Neo4jExecutor` class if needed:

```python
executor = Neo4jExecutor(
    uri="bolt://your-host:7687",
    username="your-username", 
    password="your-password"
)
```

## 📊 Output Structure

The processing function returns a dictionary with:

```python
{
    "original_text_length": int,
    "number_of_chunks": int,
    "number_of_queries": int,
    "chunks": [str, ...],
    "cypher_queries": [str, ...],
    "execution_stats": {"success": int, "errors": int}  # if execute_queries=True
}
```

## 🧪 Testing

Run the built-in example:

```bash
python neo4j_langgraph_chain.py
```

This will process a sample text and display the results.

## 🛠️ Advanced Usage

### Custom State Management

The chain uses LangGraph's state management with custom reducers:

```python
# Add custom fields to ProcessingState
class CustomProcessingState(ProcessingState):
    custom_field: str = Field(default="", description="Custom field")

# Use in your processing logic
```

### Conditional Processing

The chain includes conditional routing based on chunk completion:

```python
def should_continue_processing(state: ProcessingState) -> str:
    if state.current_chunk_index < len(state.chunks):
        return "generate_cypher"
    else:
        return "END"
```

### Memory Checkpointing

The chain uses `MemorySaver` for state persistence:

```python
# Configuration with thread ID for checkpointing
config = {"configurable": {"thread_id": "unique-session-id"}}
final_state = chain.invoke(initial_state, config)
```

## 📈 Performance Considerations

Based on project specifications:

- **LangGraph vs LangChain**: LangGraph provides better performance for mixed query scenarios through conditional routing
- **Token Optimization**: Lower temperature settings (0.1-0.3) used for consistent Cypher generation
- **State Management**: Efficient message accumulation using reducer patterns
- **Single Model Usage**: Using `llama2:7b` for both tasks reduces model loading overhead

## 🔒 Security Notes

- Default credentials are used for local development only
- In production, use environment variables for sensitive configuration
- Consider implementing authentication for the Neo4j instance

## 🐛 Troubleshooting

### Common Issues

1. **Ollama connection failed**: Ensure `ollama serve` is running and `llama2:7b` model is loaded
2. **Neo4j connection refused**: Check if Docker container is running: `docker ps`
3. **Model not found**: Verify model availability: `ollama list`
4. **Permission denied**: Ensure proper Docker permissions

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!