"""
Knowledge Commons CLI
====================

Command-line interface for the Knowledge Commons system.
"""

import os
import sys
import click
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from knowledge_commons import get_config
from knowledge_commons.core.local_storage import LocalStorage
from knowledge_commons.core.graph_db import GraphDatabase
from knowledge_commons.core.vector_db import VectorDatabase
from knowledge_commons.core.query_engine import QueryEngine
from knowledge_commons.semantic.extraction import EntityExtractor
from knowledge_commons.semantic.schema_mapper import SchemaMapper


class KnowledgeCommonsContext:
    """Context object for CLI commands, holding common resources."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the context with configuration and core components."""
        self.config = get_config(config_path)
        self.storage = None
        self.graph_db = None
        self.vector_db = None
        self.query_engine = None
        self.entity_extractor = None
        self.schema_mapper = None
    
    def init_components(self):
        """Initialize system components based on configuration."""
        # Create storage directories if they don't exist
        data_dir = Path(self.config["system"]["data_dir"])
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.storage = LocalStorage(self.config)
        self.graph_db = GraphDatabase(self.config)
        self.vector_db = VectorDatabase(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.schema_mapper = SchemaMapper(self.config)
        
        # Query engine needs the other components
        self.query_engine = QueryEngine(
            self.config,
            graph_db=self.graph_db,
            vector_db=self.vector_db,
            storage=self.storage
        )


pass_context = click.make_pass_decorator(KnowledgeCommonsContext)


@click.group()
@click.option(
    "--config", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to configuration file"
)
@click.pass_context
def cli(ctx, config):
    """Knowledge Commons - A federated personal knowledge management system."""
    # Create and initialize context
    ctx.obj = KnowledgeCommonsContext(config)


@cli.command()
@click.option(
    "--force", 
    is_flag=True, 
    help="Force initialization even if data already exists"
)
@click.option(
    "--config", 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to configuration file (overrides the global option)"
)
@click.pass_context
def init(ctx, force, config):
    """Initialize the Knowledge Commons system."""
    # If config is provided at command level, use it to override the global config
    if config:
        ctx.obj = KnowledgeCommonsContext(config)
    
    config_obj = ctx.obj.config
    data_dir = Path(config_obj["system"]["data_dir"])
    
    if data_dir.exists() and not force:
        dirs = list(data_dir.iterdir())
        if dirs:
            click.confirm(
                f"Data directory {data_dir} already exists and contains data. Reinitialize?", 
                abort=True
            )
    
    # Create required directories
    click.echo(f"Initializing Knowledge Commons in {data_dir}")
    
    # Initialize core components
    ctx.obj.init_components()
    
    # Create required directories
    for path in [
        Path(config_obj["storage"]["local"]["content_path"]),
        Path(config_obj["databases"]["graph"]["path"]),
        Path(config_obj["databases"]["vector"]["path"]),
        Path(config_obj["system"]["temp_dir"])
    ]:
        path.mkdir(exist_ok=True, parents=True)
        click.echo(f"Created directory: {path}")
    
    click.echo("Initialization complete!")


@cli.group()
@pass_context
def add(ctx):
    """Add content to the knowledge system."""
    ctx.init_components()


@add.command()
@click.argument("title")
@click.option("--content", help="Note content")
@click.option("--file", type=click.Path(exists=True), help="File containing note content")
@click.option("--tags", help="Comma-separated tags")
@pass_context
def note(ctx, title, content, file, tags):
    """Add a note to the knowledge system."""
    if content is None and file is None:
        content = click.edit()
    elif file:
        with open(file, "r") as f:
            content = f.read()
    
    if content is None:
        click.echo("No content provided. Aborting.")
        return
    
    # Prepare metadata
    metadata = {
        "title": title,
        "created": datetime.now().isoformat(),
        "type": "note"
    }
    
    if tags:
        metadata["tags"] = [tag.strip() for tag in tags.split(",")]
    
    # Extract entities
    entities = ctx.entity_extractor.extract(content)
    if entities:
        metadata["entities"] = entities
    
    # Save the note
    result = ctx.storage.save_markdown(content, metadata)
    
    # Add to graph database
    graph_entities = {}
    for entity in entities:
        # Map to ontology concepts
        mapped_entity = ctx.schema_mapper.map_entity(entity)
        # Add to graph database
        entity_id = ctx.graph_db.add_entity(mapped_entity)
        graph_entities[entity["type"]] = entity_id
    
    # Add document node
    doc_id = ctx.graph_db.add_document(
        uri=str(result["path"]),
        metadata=metadata,
        entities=graph_entities
    )
    
    # Generate embeddings and add to vector database
    embedding = ctx.entity_extractor.generate_embedding(content)
    if embedding is not None:
        ctx.vector_db.add_embedding(
            id=str(doc_id),
            embedding=embedding,
            metadata={
                "path": str(result["path"]),
                "type": "note",
                "title": title,
                "entities": [e["id"] for e in entities]
            }
        )
    
    click.echo(f"Added note: {title}")
    click.echo(f"Saved to: {result['path']}")
    if entities:
        click.echo(f"Extracted {len(entities)} entities")


@cli.command()
@click.argument("query_text")
@click.option("--limit", type=int, default=5, help="Maximum number of results")
@pass_context
def query(ctx, query_text, limit):
    """Query the knowledge system."""
    ctx.init_components()
    
    click.echo(f"Querying: {query_text}")
    
    results = ctx.query_engine.query(query_text, limit=limit)
    
    if not results["items"]:
        click.echo("No results found.")
        return
    
    click.echo(f"Found {len(results['items'])} results:")
    for i, item in enumerate(results["items"], 1):
        click.echo(f"\n{i}. {item['title']} (Score: {item['score']:.2f})")
        click.echo(f"   Type: {item['type']}")
        click.echo(f"   Path: {item['path']}")
        
        if item.get("snippet"):
            click.echo(f"   Snippet: {item['snippet']}")
        
        if item.get("entities"):
            entities = ", ".join([f"{e['type']}: {e['name']}" for e in item["entities"]])
            click.echo(f"   Entities: {entities}")


@cli.command()
@click.option("--format", type=click.Choice(["text", "json", "yaml"]), default="text")
@pass_context
def info(ctx, format):
    """Show information about the knowledge system."""
    ctx.init_components()
    
    # Collect system information
    info = {
        "version": ctx.config["system"]["version"],
        "data_directory": ctx.config["system"]["data_dir"],
        "storage": {
            "notes_count": ctx.storage.count_files_by_type("markdown"),
            "documents_count": ctx.storage.count_files_by_type("json_ld")
        },
        "graph_database": {
            "type": ctx.config["databases"]["graph"]["type"],
            "entities_count": ctx.graph_db.count_entities(),
            "relationships_count": ctx.graph_db.count_relationships()
        },
        "vector_database": {
            "type": ctx.config["databases"]["vector"]["type"],
            "vectors_count": ctx.vector_db.count_vectors()
        }
    }
    
    # Output in the requested format
    if format == "json":
        import json
        click.echo(json.dumps(info, indent=2))
    elif format == "yaml":
        click.echo(yaml.dump(info, default_flow_style=False))
    else:
        click.echo("Knowledge Commons System Information")
        click.echo("==================================")
        click.echo(f"Version: {info['version']}")
        click.echo(f"Data Directory: {info['data_directory']}")
        click.echo("\nStorage:")
        click.echo(f"  Notes: {info['storage']['notes_count']}")
        click.echo(f"  Documents: {info['storage']['documents_count']}")
        click.echo("\nGraph Database:")
        click.echo(f"  Type: {info['graph_database']['type']}")
        click.echo(f"  Entities: {info['graph_database']['entities_count']}")
        click.echo(f"  Relationships: {info['graph_database']['relationships_count']}")
        click.echo("\nVector Database:")
        click.echo(f"  Type: {info['vector_database']['type']}")
        click.echo(f"  Vectors: {info['vector_database']['vectors_count']}")


def main():
    """Entry point for the command line interface."""
    cli()


if __name__ == "__main__":
    main()