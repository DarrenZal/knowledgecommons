#!/usr/bin/env python
"""
Example: Federation with Knowledge Commons
========================================

This example demonstrates how to share knowledge between different
instances of Knowledge Commons, enabling federation and collaboration.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import knowledge_commons
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_commons import get_config
from knowledge_commons.core.local_storage import LocalStorage
from knowledge_commons.core.graph_db import GraphDatabase
from knowledge_commons.semantic.schema_mapper import SchemaMapper


class FederationExample:
    """Example class demonstrating federation between Knowledge Commons instances."""
    
    def __init__(self):
        """Initialize the example."""
        # Set up System A (source)
        print("Setting up System A (source)...")
        config_a_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
        self.config_a = get_config(str(config_a_path))
        self.storage_a = LocalStorage(self.config_a)
        self.graph_db_a = GraphDatabase(self.config_a)
        self.schema_mapper_a = SchemaMapper(self.config_a)
        
        # Set up System B (target)
        print("Setting up System B (target)...")
        # For demo, we'll use the same code but a different storage location
        config_b_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
        self.config_b = get_config(str(config_b_path))
        # Modify storage path for System B
        self.config_b["system"]["data_dir"] = "./storage_b"
        self.config_b["storage"]["local"]["content_path"] = "./storage_b/content"
        self.config_b["databases"]["graph"]["path"] = "./storage_b/graph"
        
        # Create system B directories
        Path(self.config_b["system"]["data_dir"]).mkdir(exist_ok=True, parents=True)
        Path(self.config_b["storage"]["local"]["content_path"]).mkdir(exist_ok=True, parents=True)
        Path(self.config_b["databases"]["graph"]["path"]).mkdir(exist_ok=True, parents=True)
        
        self.storage_b = LocalStorage(self.config_b)
        self.graph_db_b = GraphDatabase(self.config_b)
        self.schema_mapper_b = SchemaMapper(self.config_b)
    
    def create_sample_data(self):
        """Create sample data in System A for federation."""
        print("\nCreating sample data in System A...")
        
        # Create a person entity
        person = {
            "id": "jane-doe",
            "type": "Person",
            "name": "Jane Doe",
            "email": "jane@example.com",
            "description": "AI researcher specializing in knowledge graphs and federation."
        }
        
        # Add to graph database
        print("Adding person to graph database...")
        person_id = self.graph_db_a.add_entity(person)
        print(f"  Person ID: {person_id}")
        
        # Create an organization entity
        organization = {
            "id": "knowledge-labs",
            "type": "Organization",
            "name": "Knowledge Labs",
            "description": "Research lab focused on knowledge representation and sharing.",
            "website": "https://knowledgelabs.example.org"
        }
        
        # Add to graph database
        print("Adding organization to graph database...")
        org_id = self.graph_db_a.add_entity(organization)
        print(f"  Organization ID: {org_id}")
        
        # Create a project entity
        project = {
            "id": "federated-knowledge",
            "type": "Project",
            "name": "Federated Knowledge Commons",
            "description": "A project to create standards for federated knowledge sharing.",
            "startDate": "2025-01-15T00:00:00Z"
        }
        
        # Add to graph database
        print("Adding project to graph database...")
        project_id = self.graph_db_a.add_entity(project)
        print(f"  Project ID: {project_id}")
        
        # Create relationships
        print("Adding relationships...")
        
        # Person works for Organization
        self.graph_db_a.add_relationship(
            source="Person:jane-doe",
            target="Organization:knowledge-labs",
            relation_type="memberOf"
        )
        
        # Person works on Project
        self.graph_db_a.add_relationship(
            source="Person:jane-doe",
            target="Project:federated-knowledge",
            relation_type="worksOn"
        )
        
        # Organization sponsors Project
        self.graph_db_a.add_relationship(
            source="Organization:knowledge-labs",
            target="Project:federated-knowledge",
            relation_type="sponsors"
        )
        
        # Create a note document
        note_content = """
        # Federated Knowledge Commons Proposal
        
        ## Overview
        
        The Federated Knowledge Commons project aims to create standards and protocols
        for sharing knowledge across different systems and communities. This proposal
        outlines the key components and architecture.
        
        ## Key Components
        
        1. **Standard Data Formats** - RDF, JSON-LD, and Markdown with YAML frontmatter
        2. **Federation Protocol** - Based on ActivityPub and SPARQL Federation
        3. **Access Control** - Granular permissions for different sharing contexts
        4. **Schema Registry** - Mapping between different ontologies
        
        ## Timeline
        
        - Q2 2025: Protocol specification
        - Q3 2025: Reference implementation
        - Q4 2025: First federation partners
        """
        
        # Prepare metadata
        metadata = {
            "title": "Federated Knowledge Commons Proposal",
            "type": "note",
            "created": datetime.now().isoformat(),
            "tags": ["federation", "proposal", "standards", "knowledge-commons"],
            "project": "federated-knowledge",
            "author": "jane-doe"
        }
        
        # Save the note
        print("Adding note document...")
        result = self.storage_a.save_markdown(note_content, metadata)
        print(f"  Saved to: {result['path']}")
        
        # Add document to graph database
        doc_id = self.graph_db_a.add_document(
            uri=str(result["path"]),
            metadata=metadata,
            entities={
                "Person": "jane-doe",
                "Project": "federated-knowledge",
                "Organization": "knowledge-labs"
            }
        )
        print(f"  Document ID: {doc_id}")
        
        # Save the data
        self.graph_db_a.save()
        
        return {
            "person_id": person_id,
            "org_id": org_id,
            "project_id": project_id,
            "doc_id": doc_id,
            "doc_path": result["path"]
        }
    
    def export_knowledge_package(self, entities):
        """Export a knowledge package from System A."""
        print("\nExporting knowledge package from System A...")
        
        # Create a new graph for the export
        export_graph = self.graph_db_a.graph.query("""
            CONSTRUCT {
                ?s ?p ?o .
            }
            WHERE {
                {
                    # Include the person entity and its properties
                    ?s a <http://xmlns.com/foaf/0.1/Person> .
                    ?s ?p ?o .
                    FILTER(CONTAINS(STR(?s), "jane-doe"))
                } UNION {
                    # Include the organization entity and its properties
                    ?s a <http://xmlns.com/foaf/0.1/Organization> .
                    ?s ?p ?o .
                    FILTER(CONTAINS(STR(?s), "knowledge-labs"))
                } UNION {
                    # Include the project entity and its properties
                    ?s a ?type .
                    ?s ?p ?o .
                    FILTER(CONTAINS(STR(?s), "federated-knowledge"))
                } UNION {
                    # Include relationships between these entities
                    ?s ?p ?o .
                    FILTER(
                        (CONTAINS(STR(?s), "jane-doe") && CONTAINS(STR(?o), "knowledge-labs")) ||
                        (CONTAINS(STR(?s), "jane-doe") && CONTAINS(STR(?o), "federated-knowledge")) ||
                        (CONTAINS(STR(?s), "knowledge-labs") && CONTAINS(STR(?o), "federated-knowledge"))
                    )
                }
            }
        """)
        
        # Count triples in export
        print(f"  Exported {len(export_graph)} triples")
        
        # Create the knowledge package
        package = {
            "metadata": {
                "creator": "System A",
                "created": datetime.now().isoformat(),
                "description": "Sample knowledge package for federation example",
                "format": "RDF"
            },
            "graph": export_graph.serialize(format="json-ld")
        }
        
        # Export the document content
        if entities.get("doc_path"):
            try:
                doc_data = self.storage_a.load_markdown(entities["doc_path"])
                package["documents"] = [{
                    "id": entities.get("doc_id", "unknown"),
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"]
                }]
            except Exception as e:
                print(f"  Warning: Could not load document: {e}")
        
        # Save the package to a file
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        
        export_file = export_dir / "knowledge_package.json"
        with open(export_file, "w") as f:
            json.dump(package, f, indent=2)
        
        print(f"  Package saved to: {export_file}")
        
        return package
    
    def import_knowledge_package(self, package):
        """Import a knowledge package into System B."""
        print("\nImporting knowledge package into System B...")
        
        # Parse the incoming graph
        import rdflib
        incoming_graph = rdflib.Graph()
        incoming_graph.parse(data=package["graph"], format="json-ld")
        
        # Load into System B's graph
        print(f"  Importing {len(incoming_graph)} triples...")
        
        for triple in incoming_graph:
            self.graph_db_b.graph.add(triple)
        
        # Import documents if included
        if "documents" in package:
            for doc in package["documents"]:
                print(f"  Importing document: {doc.get('id', 'unknown')}")
                
                # Save the document
                result = self.storage_b.save_markdown(
                    doc["content"],
                    doc["metadata"]
                )
                print(f"    Saved to: {result['path']}")
                
                # Add document to graph database if not already there
                self.graph_db_b.add_document(
                    uri=str(result["path"]),
                    metadata=doc["metadata"]
                )
        
        # Save the updated graph
        self.graph_db_b.save()
        
        print("  Import complete")
    
    def verify_import(self):
        """Verify that data was correctly imported into System B."""
        print("\nVerifying import in System B...")
        
        # Check for the person entity
        person = self.graph_db_b.get_entity("Person", "jane-doe")
        if person:
            print(f"  Found person: {person.get('name')}")
        else:
            print("  Person not found!")
        
        # Check for the organization entity
        org = self.graph_db_b.get_entity("Organization", "knowledge-labs")
        if org:
            print(f"  Found organization: {org.get('name')}")
        else:
            print("  Organization not found!")
        
        # Check for the project entity
        project = self.graph_db_b.get_entity("Project", "federated-knowledge")
        if project:
            print(f"  Found project: {project.get('name')}")
        else:
            print("  Project not found!")
        
        # Check for relationships - run a SPARQL query
        query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX kc: <https://knowledge-commons.example/ontology#>
        
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
            FILTER(
                (CONTAINS(STR(?s), "jane-doe") && CONTAINS(STR(?o), "knowledge-labs")) ||
                (CONTAINS(STR(?s), "jane-doe") && CONTAINS(STR(?o), "federated-knowledge")) ||
                (CONTAINS(STR(?s), "knowledge-labs") && CONTAINS(STR(?o), "federated-knowledge"))
            )
        }
        """
        
        results = self.graph_db_b.query(query)
        print(f"  Found {len(results)} relationships between entities")
        
        # Check for imported document
        note_files = list(Path(self.config_b["storage"]["local"]["content_path"]).glob("**/*proposal*.md"))
        if note_files:
            print(f"  Found {len(note_files)} imported document files")
            for file_path in note_files:
                print(f"    - {file_path}")
        else:
            print("  No imported documents found")
        
        # Demonstrate SPARQL federation (simulated)
        print("\nDemonstrating simulated SPARQL federation...")
        print("  In a real federation scenario, we would use the SERVICE keyword to query across systems")
        
        # We'll simulate this by querying both graphs and combining results
        query_a = """
        SELECT ?name ?org WHERE {
            ?person a <http://xmlns.com/foaf/0.1/Person> .
            ?person <http://xmlns.com/foaf/0.1/name> ?name .
            ?person ?rel ?organization .
            ?organization a <http://xmlns.com/foaf/0.1/Organization> .
            ?organization <http://xmlns.com/foaf/0.1/name> ?org .
        }
        """
        
        results_a = self.graph_db_a.query(query_a)
        results_b = self.graph_db_b.query(query_a)
        
        print("  Results from System A:")
        for result in results_a:
            print(f"    - {result['name']} works at {result['org']}")
        
        print("  Results from System B:")
        for result in results_b:
            print(f"    - {result['name']} works at {result['org']}")


def main():
    """Run the federation example."""
    print("Knowledge Commons Example: Federation")
    print("===================================")
    
    # Create the example instance
    example = FederationExample()
    
    # Create sample data in System A
    entities = example.create_sample_data()
    
    # Export knowledge package from System A
    package = example.export_knowledge_package(entities)
    
    # Import knowledge package into System B
    example.import_knowledge_package(package)
    
    # Verify the import
    example.verify_import()
    
    print("\nFederation example completed!")


if __name__ == "__main__":
    main()