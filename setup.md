# Setting Up Knowledge Commons

This guide will walk you through setting up the Knowledge Commons project on your local machine.

## Prerequisites

- Python 3.9 or higher
- Git
- Optional: Docker (for running database services)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/knowledge-commons.git
cd knowledge-commons
```

### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Set Up Configuration

```bash
# Copy the default configuration
cp config/default_config.yaml config/my_config.yaml
```

Edit `config/my_config.yaml` to customize your setup. At minimum, you should update:

- `user.id` - Your identifier
- `user.name` - Your name
- `system.data_dir` - Where your knowledge data will be stored

### 5. Initialize the System

```bash
# Create necessary directories and initial data structures
knowledge-commons init --config config/my_config.yaml
```

### 6. Running the Examples

The project includes several example scripts to help you get started:

```bash
# Add a sample note
python examples/add_note.py

# Query the knowledge base
python examples/query_knowledge.py

# Test federation capabilities
python examples/federation_example.py
```

## Optional: Using External Services

### Vector Database Options

By default, Knowledge Commons uses local file-based databases. For better performance, you can use:

#### Using Qdrant

```bash
# Run Qdrant with Docker
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

# Update your config to use remote Qdrant
# In config/my_config.yaml:
# databases:
#   vector:
#     type: "qdrant"
#     remote:
#       enabled: true
#       url: "http://localhost:6333"
```

#### Using ChromaDB

```bash
# Install ChromaDB
pip install chromadb

# Update your config to use ChromaDB
# In config/my_config.yaml:
# databases:
#   vector:
#     type: "chroma"
```

### Graph Database Options

For advanced graph capabilities, you can use:

#### Using Apache Jena Fuseki

```bash
# Run Fuseki with Docker
docker run -p 3030:3030 -v $(pwd)/fuseki_data:/fuseki stain/jena-fuseki

# Create a dataset named "knowledge" in the Fuseki web interface
# Then update your config:
# databases:
#   graph:
#     type: "jena"
#     remote:
#       enabled: true
#       url: "http://localhost:3030/knowledge"
```

#### Using TerminusDB

```bash
# Run TerminusDB with Docker
docker run -p 6363:6363 -v $(pwd)/terminusdb_data:/app/terminusdb/storage terminusdb/terminusdb-server:dev

# Update your config accordingly
```

## LLM Integration

For entity extraction and embedding generation, you can use:

### Using Anthropic Claude

```bash
# Install the Anthropic package
pip install anthropic

# Add your API key to your environment
export ANTHROPIC_API_KEY="your-api-key"

# Or update your config:
# llm:
#   provider: "anthropic"
#   model: "claude-3-opus-20240229"
#   api_key: "your-api-key"
```

### Using OpenAI

```bash
# Install the OpenAI package
pip install openai

# Add your API key to your environment
export OPENAI_API_KEY="your-api-key"

# Update your config:
# llm:
#   provider: "openai"
#   model: "gpt-4-turbo"
#   api_key: "your-api-key"
```

### Using Local Models

```bash
# Install sentence-transformers for local embeddings
pip install sentence-transformers

# Update your config:
# llm:
#   embedding_model: "all-mpnet-base-v2"
```

## Customizing Schemas

The system uses schema mappings defined in `config/schema_mappings.yaml`. You can customize these mappings to support additional entity types or properties.

## Troubleshooting

- **Import Errors**: Ensure you've installed all dependencies with `pip install -e .`
- **Configuration Issues**: Check that your config file has valid YAML syntax
- **Database Connection Problems**: Verify that any external databases are running and accessible
- **LLM API Errors**: Check that your API keys are correctly configured

For more help, please open an issue on the GitHub repository.