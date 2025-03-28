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

# Install embedding model dependencies
pip install transformers torch sentence-transformers
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:

```
ANTHROPIC_API_KEY=your-anthropic-api-key
# Add other API keys as needed
```

Make sure to add `.env` to your `.gitignore` file to prevent accidentally committing sensitive information.

### 5. Set Up Configuration

```bash
# Copy the default configuration
cp config/default_config.yaml config/my_config.yaml
```

Edit `config/my_config.yaml` to customize your setup. Here's a recommended configuration:

```yaml
# User and System Identification
user:
  id: "your-email@example.com"
  name: "Your Name"

# System Configuration
system:
  name: "Knowledge Commons"
  version: "0.1.0"
  data_dir: "./storage"
  temp_dir: "./tmp"

# LLM Integration
llm:
  enabled: true
  provider: "anthropic"
  model: "claude-3-opus-20240229"
  api_key: "your-api-key-here"  # Replace with your actual API key or use .env
  
  # Embedding configuration - using Stella for high-quality local embeddings
  embeddings:
    provider: "huggingface"
    model: "neulab/stella-400m-v5"
```

### 6. Initialize the System

```bash
# Create necessary directories and initial data structures
knowledge-commons init --config config/my_config.yaml
```

### 7. Using Stella for Embeddings

Knowledge Commons is configured to use the Stella model for generating embeddings. Stella is a high-performance open-source embedding model that offers several advantages:

- **High Quality**: Performs very well compared to commercial options
- **Open Source**: MIT license allows free use for any purpose
- **Multilingual**: Works with both English and other languages
- **Local Processing**: No API calls or external dependencies required
- **Efficient Size**: The 400M parameter version balances performance and resource usage

When you first run a command that needs embeddings, the system will automatically download the model from Hugging Face. The model is about 800MB in size, so the initial download may take a few minutes depending on your internet connection.

### 8. Running the Examples

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

## Alternative Embedding Models

While Stella is configured as the default embedding model, you can use other options:

### Using OpenAI Embeddings

```bash
# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your-api-key" >> .env

# Update your config:
# llm:
#   embeddings:
#     provider: "openai"
#     model: "text-embedding-3-small"
```

### Using Voyage AI Embeddings (Recommended by Anthropic)

```bash
# Add your Voyage API key to .env
echo "VOYAGE_API_KEY=your-api-key" >> .env

# Install the Voyage client
pip install voyageai

# Update your config:
# llm:
#   embeddings:
#     provider: "voyage"
#     model: "voyage-3-lite"  # or voyage-3-large for higher quality
```

### Using Sentence Transformers

For other alternative models:

```bash
# Update your config:
# llm:
#   embeddings:
#     provider: "sentence_transformers"
#     model: "all-mpnet-base-v2"  # or another model name
```

## Customizing Schemas

The system uses schema mappings defined in `config/schema_mappings.yaml`. You can customize these mappings to support additional entity types or properties.

## Troubleshooting

- **Import Errors**: Ensure you've installed all dependencies with `pip install -e .`
- **Configuration Issues**: Check that your config file has valid YAML syntax
- **Database Connection Problems**: Verify that any external databases are running and accessible
- **LLM API Errors**: Check that your API keys are correctly set in the `.env` file
- **Environment Variables**: If API keys aren't being loaded, make sure python-dotenv is installed and the `.env` file is in the correct location
- **Embedding Model Errors**: If you encounter issues with the Stella model, try using `sentence_transformers` as a fallback

For more help, please open an issue on the GitHub repository.