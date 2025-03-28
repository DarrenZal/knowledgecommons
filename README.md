# Knowledge Commons

A federated personal knowledge management system designed for sharing and collaboration.

## Overview

Knowledge Commons is a system that helps you manage personal knowledge while enabling selective sharing with communities. It combines graph databases, vector search, and semantic technologies to create an interoperable knowledge ecosystem.

Key features:
- Store and manage notes, documents, and structured data
- Link entities and concepts in a knowledge graph
- Use AI to enhance retrieval and organization
- Selectively share knowledge with different groups
- Federate with other knowledge systems

## Architecture

Knowledge Commons uses a hybrid architecture:
- **Graph Database**: For storing entities and relationships
- **Vector Database**: For semantic search and retrieval
- **Local Storage**: For original content in standard formats
- **Semantic Layer**: For entity extraction and schema mapping
- **Federation Layer**: For sharing with other systems

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for running databases)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/knowledge-commons.git
cd knowledge-commons

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

Copy the default configuration file and edit it:

```bash
cp config/default_config.yaml config/my_config.yaml
```

Edit `config/my_config.yaml` to set up your storage locations, database connections, and other preferences.

### Running

```bash
# Initialize the system
knowledge-commons init --config config/my_config.yaml

# Add a note
knowledge-commons add note "My first note" --content "This is a test note" --tags test,example

# Query the system
knowledge-commons query "What notes do I have about tests?"
```

## Current Status

This project is in active development. Current capabilities include:
- Local storage of markdown files with YAML frontmatter
- Basic graph database integration
- Simple entity extraction
- Initial vector search capabilities

Upcoming features:
- Advanced federation protocols
- Enhanced AI integration
- Web interface
- Mobile apps

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.