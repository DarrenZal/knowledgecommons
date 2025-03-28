"""
Graph Database Module
====================

Handles interaction with graph databases for Knowledge Commons.
Supports RDFLib local graph storage and can be extended to support
external graph databases like TerminusDB, Jena, etc.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import rdflib
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF, SKOS
from rdflib.namespace import DC, DCTERMS 

class GraphDatabase:
    """
    Manages graph database interactions for Knowledge Commons.
    Uses RDFLib as the default implementation with option to extend.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the graph database manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.db_type = config["databases"]["graph"]["type"]
        self.db_path = Path(config["databases"]["graph"]["path"])
        self.db_path.mkdir(exist_ok=True, parents=True)
        
        # Define namespaces
        self.SCHEMA = Namespace("http://schema.org/")
        self.KC = Namespace(config["schema"]["base_uri"])
        
        # Initialize the graph
        self.graph = self._init_graph()
        
        # Bind common namespaces
        self._bind_namespaces()
    
    def _init_graph(self) -> Graph:
        """
        Initialize the graph database based on configuration.
        
        Returns:
            Initialized RDF graph
        """
        if self.db_type == "rdflib":
            # Create a local RDF graph
            graph = Graph()
            
            # Load existing data if available
            graph_file = self.db_path / "knowledge_graph.ttl"
            if graph_file.exists():
                graph.parse(str(graph_file), format=self.config["databases"]["graph"]["format"])
            
            return graph
        elif self.db_type in ["terminusdb", "jena", "neo4j"]:
            # Placeholder for future implementations
            raise NotImplementedError(f"Graph database type '{self.db_type}' not yet implemented")
        else:
            raise ValueError(f"Unsupported graph database type: {self.db_type}")
    
    def _bind_namespaces(self):
        """Bind common namespaces to the graph."""
        self.graph.bind("schema", self.SCHEMA)
        self.graph.bind("kc", self.KC)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("skos", SKOS)
        self.graph.bind("dc", DC)
        
        # Add any custom namespaces from config
        for prefix, uri in self.config["schema"].get("default_context", {}).items():
            if prefix not in ["schema", "rdf", "rdfs", "xsd", "dc", "foaf", "skos", "kc"]:
                self.graph.bind(prefix, Namespace(uri))
    
    def save(self):
        """Save the graph to persistent storage."""
        if self.db_type == "rdflib":
            graph_file = self.db_path / "knowledge_graph.ttl"
            self.graph.serialize(destination=str(graph_file), format=self.config["databases"]["graph"]["format"])
    
    def add_entity(self, entity: Dict[str, Any]) -> str:
        """
        Add an entity to the graph database.
        
        Args:
            entity: Dictionary containing entity data
            
        Returns:
            String identifier for the entity
        """
        # Extract basic entity information
        entity_type = entity.get("type", "Thing")
        entity_name = entity.get("name", "")
        entity_id = entity.get("id", str(uuid.uuid4()))
        
        # Create a URI for the entity
        entity_uri = self._create_entity_uri(entity_type, entity_id)
        
        # Add standard RDF typing
        if entity_type == "Person":
            self.graph.add((entity_uri, RDF.type, FOAF.Person))
            if entity_name:
                self.graph.add((entity_uri, FOAF.name, Literal(entity_name)))
        elif entity_type == "Organization":
            self.graph.add((entity_uri, RDF.type, FOAF.Organization))
            if entity_name:
                self.graph.add((entity_uri, FOAF.name, Literal(entity_name)))
        elif entity_type == "Note":
            self.graph.add((entity_uri, RDF.type, self.KC.Note))
            if entity_name:
                self.graph.add((entity_uri, DC.title, Literal(entity_name)))
        elif entity_type == "Event":
            self.graph.add((entity_uri, RDF.type, self.SCHEMA.Event))
            if entity_name:
                self.graph.add((entity_uri, self.SCHEMA.name, Literal(entity_name)))
        elif entity_type == "Task":
            self.graph.add((entity_uri, RDF.type, self.KC.Task))
            if entity_name:
                self.graph.add((entity_uri, RDFS.label, Literal(entity_name)))
        elif entity_type == "Project":
            self.graph.add((entity_uri, RDF.type, self.KC.Project))
            if entity_name:
                self.graph.add((entity_uri, RDFS.label, Literal(entity_name)))
        elif entity_type == "Concept":
            self.graph.add((entity_uri, RDF.type, SKOS.Concept))
            if entity_name:
                self.graph.add((entity_uri, SKOS.prefLabel, Literal(entity_name)))
        else:
            # Generic entity
            self.graph.add((entity_uri, RDF.type, self.SCHEMA.Thing))
            if entity_name:
                self.graph.add((entity_uri, self.SCHEMA.name, Literal(entity_name)))
        
        # Add additional properties
        self._add_entity_properties(entity_uri, entity)
        
        # Save changes
        self.save()
        
        return entity_id
    
    def _create_entity_uri(self, entity_type: str, entity_id: str) -> URIRef:
        """
        Create a URI for an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            
        Returns:
            URIRef for the entity
        """
        # Convert type to lowercase for URI
        type_lower = entity_type.lower()
        
        # Create the URI
        return URIRef(f"{self.KC}{type_lower}/{entity_id}")
    
    def _add_entity_properties(self, entity_uri: URIRef, entity: Dict[str, Any]):
        """
        Add properties to an entity in the graph.
        
        Args:
            entity_uri: URI of the entity
            entity: Dictionary containing entity properties
        """
        # Add other properties based on entity type
        entity_type = entity.get("type", "Thing")
        
        # Load schema mappings for the entity type
        schema_mappings_file = Path(self.config["schema"]["mappings_file"])
        schema_mappings = {}
        
        if schema_mappings_file.exists():
            import yaml
            with open(schema_mappings_file, "r") as f:
                all_mappings = yaml.safe_load(f)
                schema_mappings = all_mappings.get(entity_type, {}).get("properties", {})
        
        # Add all properties from the entity dictionary
        for key, value in entity.items():
            if key in ["id", "type", "name"]:
                # These are handled separately
                continue
            
            # Skip None values
            if value is None:
                continue
                
            # Check if we have a mapping for this property
            if key in schema_mappings:
                # Use the RDF predicate from the mapping
                rdf_predicate = schema_mappings[key].get("rdf")
                if rdf_predicate:
                    predicate = URIRef(rdf_predicate)
                else:
                    # Fall back to KC namespace
                    predicate = self.KC[key]
                    
                # Handle different value types
                if schema_mappings[key].get("type") == "datetime":
                    # Convert to XSD datetime
                    obj = Literal(value, datatype=XSD.dateTime)
                elif schema_mappings[key].get("type") == "list":
                    # For list values, add multiple triples
                    for item in value:
                        self.graph.add((entity_uri, predicate, Literal(item)))
                    continue
                elif schema_mappings[key].get("type") == "reference":
                    # For references to other entities
                    ref_type = value.get("type", "Thing")
                    ref_id = value.get("id")
                    if ref_id:
                        obj = self._create_entity_uri(ref_type, ref_id)
                    else:
                        # Skip if no ID
                        continue
                else:
                    # Default to string literal
                    obj = Literal(value)
            else:
                # No mapping, use a generic approach
                predicate = self.KC[key]
                obj = Literal(value)
            
            # Add the triple
            self.graph.add((entity_uri, predicate, obj))
    
    def add_document(self, uri: str, metadata: Dict[str, Any], entities: Dict[str, str] = None) -> str:
        """
        Add a document to the graph with references to entities.
        
        Args:
            uri: URI or path to the document
            metadata: Document metadata
            entities: Dictionary mapping entity types to entity IDs
            
        Returns:
            String identifier for the document
        """
        # Generate a document ID
        doc_id = str(uuid.uuid4())
        doc_uri = URIRef(f"{self.KC}document/{doc_id}")
        
        # Add basic document information
        self.graph.add((doc_uri, RDF.type, self.KC.Document))
        self.graph.add((doc_uri, self.KC.uri, Literal(uri)))
        
        # Add document metadata
        for key, value in metadata.items():
            if key == "title":
                self.graph.add((doc_uri, DC.title, Literal(value)))
            elif key == "created":
                self.graph.add((doc_uri, DCTERMS.created, Literal(value)))
            elif key == "type":
                self.graph.add((doc_uri, DC.type, Literal(value)))
            elif key == "tags" and isinstance(value, (list, tuple)):
                for tag in value:
                    tag_uri = URIRef(f"{self.KC}tag/{tag}")
                    self.graph.add((tag_uri, RDF.type, self.KC.Tag))
                    self.graph.add((tag_uri, RDFS.label, Literal(tag)))
                    self.graph.add((doc_uri, self.KC.hasTag, tag_uri))
            else:
                # Generic property
                self.graph.add((doc_uri, self.KC[key], Literal(str(value))))
        
        # Link document to entities
        if entities:
            for entity_type, entity_id in entities.items():
                entity_uri = self._create_entity_uri(entity_type, entity_id)
                self.graph.add((doc_uri, self.KC.mentions, entity_uri))
        
        # Save changes
        self.save()
        
        return doc_id
    
    def add_relationship(self, 
                        source: str, 
                        target: str, 
                        relation_type: str,
                        metadata: Dict[str, Any] = None) -> str:
        """
        Add a relationship between entities.
        
        Args:
            source: Source entity identifier (format: "type:id")
            target: Target entity identifier (format: "type:id")
            relation_type: Type of relationship
            metadata: Optional relationship metadata
            
        Returns:
            String identifier for the relationship
        """
        # Parse entity identifiers
        source_parts = source.split(":", 1)
        target_parts = target.split(":", 1)
        
        if len(source_parts) != 2 or len(target_parts) != 2:
            raise ValueError("Entity identifiers must be in format 'type:id'")
        
        source_type, source_id = source_parts
        target_type, target_id = target_parts
        
        # Special case for current document
        if source_id == "current":
            # TODO: Implement a way to get the current document ID
            raise NotImplementedError("Reference to 'current' document not yet implemented")
        
        # Get the entity URIs
        source_uri = self._create_entity_uri(source_type, source_id)
        target_uri = self._create_entity_uri(target_type, target_id)
        
        # Load relationship mappings
        schema_mappings_file = Path(self.config["schema"]["mappings_file"])
        rel_predicate = None
        
        if schema_mappings_file.exists():
            import yaml
            with open(schema_mappings_file, "r") as f:
                all_mappings = yaml.safe_load(f)
                relationships = all_mappings.get("Relationships", {})
                
                if relation_type in relationships:
                    rel_mapping = relationships[relation_type]
                    if "rdf" in rel_mapping:
                        rel_predicate = URIRef(rel_mapping["rdf"])
        
        # If no mapping found, use KC namespace
        if rel_predicate is None:
            rel_predicate = self.KC[relation_type]
        
        # Add the relationship triple
        self.graph.add((source_uri, rel_predicate, target_uri))
        
        # If metadata provided, create a reified statement
        if metadata:
            # Create a unique ID for the relationship
            rel_id = str(uuid.uuid4())
            rel_uri = URIRef(f"{self.KC}relationship/{rel_id}")
            
            # Create a reified statement
            self.graph.add((rel_uri, RDF.type, RDF.Statement))
            self.graph.add((rel_uri, RDF.subject, source_uri))
            self.graph.add((rel_uri, RDF.predicate, rel_predicate))
            self.graph.add((rel_uri, RDF.object, target_uri))
            
            # Add metadata
            for key, value in metadata.items():
                self.graph.add((rel_uri, self.KC[key], Literal(str(value))))
                
            # Save changes
            self.save()
            
            return rel_id
        
        # Save changes
        self.save()
        
        # Return a simple identifier for the relationship
        return f"{source}-{relation_type}-{target}"
    
    def get_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity from the graph database.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            
        Returns:
            Dictionary containing entity data, or None if not found
        """
        # Get the entity URI
        entity_uri = self._create_entity_uri(entity_type, entity_id)
        
        # Check if the entity exists
        if (entity_uri, RDF.type, None) not in self.graph:
            return None
        
        # Build the entity dictionary
        entity = {
            "id": entity_id,
            "type": entity_type
        }
        
        # Get the entity name/label based on type
        if entity_type == "Person":
            name_predicates = [FOAF.name]
        elif entity_type == "Organization":
            name_predicates = [FOAF.name]
        elif entity_type == "Note":
            name_predicates = [DC.title]
        elif entity_type == "Event" or entity_type == "Thing":
            name_predicates = [self.SCHEMA.name]
        elif entity_type == "Task" or entity_type == "Project":
            name_predicates = [RDFS.label]
        elif entity_type == "Concept":
            name_predicates = [SKOS.prefLabel]
        else:
            name_predicates = [RDFS.label, self.SCHEMA.name, DC.title, FOAF.name]
        
        # Try each name predicate
        for predicate in name_predicates:
            for obj in self.graph.objects(entity_uri, predicate):
                entity["name"] = str(obj)
                break
            if "name" in entity:
                break
        
        # Get all other properties
        for predicate, obj in self.graph.predicate_objects(entity_uri):
            # Skip already processed properties
            if predicate in [RDF.type] or str(predicate) in [str(p) for p in name_predicates]:
                continue
            
            # Get the property name
            if str(predicate).startswith(str(self.KC)):
                # Property in KC namespace
                prop_name = str(predicate)[len(str(self.KC)):]
            elif str(predicate).startswith(str(self.SCHEMA)):
                # Property in Schema.org namespace
                prop_name = str(predicate)[len(str(self.SCHEMA)):]
            elif str(predicate).startswith(str(FOAF)):
                # Property in FOAF namespace
                prop_name = str(predicate)[len(str(FOAF)):]
            elif str(predicate).startswith(str(DC)):
                # Property in DC namespace
                prop_name = str(predicate)[len(str(DC)):]
            else:
                # Unknown namespace, use full URI
                prop_name = str(predicate)
            
            # Convert the object to an appropriate Python value
            if isinstance(obj, Literal):
                # Literal value
                value = obj.toPython()
            elif isinstance(obj, URIRef):
                # Reference to another entity
                if str(obj).startswith(str(self.KC)):
                    # Reference to a KC entity
                    ref_parts = str(obj)[len(str(self.KC)):].split('/', 1)
                    if len(ref_parts) == 2:
                        ref_type, ref_id = ref_parts
                        value = {
                            "type": ref_type.capitalize(),
                            "id": ref_id
                        }
                    else:
                        # Unknown reference format
                        value = str(obj)
                else:
                    # External reference
                    value = str(obj)
            else:
                # Other (e.g., BNode)
                value = str(obj)
            
            # Add to entity properties
            if prop_name in entity:
                # Property already exists, convert to list if needed
                if not isinstance(entity[prop_name], list):
                    entity[prop_name] = [entity[prop_name]]
                entity[prop_name].append(value)
            else:
                entity[prop_name] = value
        
        return entity
    
    def search_entities(self, 
                       entity_type: Optional[str] = None, 
                       properties: Dict[str, Any] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities in the graph database.
        
        Args:
            entity_type: Optional type of entity to search for
            properties: Optional properties to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of entity dictionaries
        """
        results = []
        
        # Build SPARQL query
        if entity_type:
            # Convert to proper RDF type URI based on entity type
            if entity_type == "Person":
                type_uri = FOAF.Person
            elif entity_type == "Organization":
                type_uri = FOAF.Organization
            elif entity_type == "Note":
                type_uri = self.KC.Note
            elif entity_type == "Event":
                type_uri = self.SCHEMA.Event
            elif entity_type == "Task":
                type_uri = self.KC.Task
            elif entity_type == "Project":
                type_uri = self.KC.Project
            elif entity_type == "Concept":
                type_uri = SKOS.Concept
            else:
                type_uri = self.SCHEMA.Thing
            
            # Query for entities of this type
            entities = list(self.graph.subjects(RDF.type, type_uri))
        else:
            # Get all entities with a type
            entities = list(self.graph.subjects(RDF.type, None))
        
        # Filter by properties if provided
        if properties:
            filtered_entities = []
            for entity_uri in entities:
                match = True
                
                for prop_name, prop_value in properties.items():
                    # Determine the predicate URI
                    if prop_name == "name":
                        # Use appropriate name predicate based on entity types
                        entity_types = list(self.graph.objects(entity_uri, RDF.type))
                        
                        if FOAF.Person in entity_types or FOAF.Organization in entity_types:
                            predicate = FOAF.name
                        elif self.KC.Note in entity_types:
                            predicate = DC.title
                        elif self.SCHEMA.Event in entity_types or self.SCHEMA.Thing in entity_types:
                            predicate = self.SCHEMA.name
                        elif self.KC.Task in entity_types or self.KC.Project in entity_types:
                            predicate = RDFS.label
                        elif SKOS.Concept in entity_types:
                            predicate = SKOS.prefLabel
                        else:
                            # Try multiple predicates
                            predicates = [RDFS.label, self.SCHEMA.name, DC.title, FOAF.name]
                            found = False
                            
                            for pred in predicates:
                                if (entity_uri, pred, None) in self.graph:
                                    found = any(
                                        prop_value.lower() in str(obj).lower() 
                                        for obj in self.graph.objects(entity_uri, pred)
                                    )
                                    if found:
                                        break
                            
                            if not found:
                                match = False
                            
                            # Continue to next property
                            continue
                    else:
                        # Use KC namespace for other properties
                        predicate = self.KC[prop_name]
                    
                    # Check if the property matches
                    if not any(
                        prop_value.lower() in str(obj).lower() 
                        for obj in self.graph.objects(entity_uri, predicate)
                    ):
                        match = False
                        break
                
                if match:
                    filtered_entities.append(entity_uri)
            
            entities = filtered_entities
        
        # Convert entity URIs to entity dictionaries
        for entity_uri in entities[:limit]:
            # Extract entity type and ID from URI
            uri_str = str(entity_uri)
            if uri_str.startswith(str(self.KC)):
                parts = uri_str[len(str(self.KC)):].split('/', 1)
                if len(parts) == 2:
                    ent_type, ent_id = parts
                    # Capitalize entity type
                    ent_type = ent_type.capitalize()
                    
                    # Get the full entity
                    entity = self.get_entity(ent_type, ent_id)
                    if entity:
                        results.append(entity)
        
        return results
    
    def find_related_documents(self, entities: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find documents related to specified entities.
        
        Args:
            entities: List of entity dictionaries
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries
        """
        results = []
        
        # Convert entities to URIs
        entity_uris = []
        for entity in entities:
            entity_type = entity.get("type", "Thing")
            entity_id = entity.get("id")
            
            if entity_id:
                entity_uri = self._create_entity_uri(entity_type, entity_id)
                entity_uris.append(entity_uri)
        
        # Find documents that mention these entities
        document_uris = set()
        for entity_uri in entity_uris:
            for doc_uri in self.graph.subjects(self.KC.mentions, entity_uri):
                if (doc_uri, RDF.type, self.KC.Document) in self.graph:
                    document_uris.add(doc_uri)
        
        # Convert document URIs to document dictionaries
        for doc_uri in list(document_uris)[:limit]:
            doc = {}
            
            # Get the document ID
            doc_id = str(doc_uri).split('/')[-1]
            doc["id"] = doc_id
            
            # Get the document URI
            for obj in self.graph.objects(doc_uri, self.KC.uri):
                doc["path"] = str(obj)
                break
            
            # Get the document title
            for obj in self.graph.objects(doc_uri, DC.title):
                doc["title"] = str(obj)
                break
            
            # Get the document type
            for obj in self.graph.objects(doc_uri, DC.type):
                doc["type"] = str(obj)
                break
            
            # Get the document creation date
            for obj in self.graph.objects(doc_uri, DCTERMS.created):
                doc["created"] = str(obj)
                break
            
            # Get mentions
            doc["entities"] = []
            for obj in self.graph.objects(doc_uri, self.KC.mentions):
                # Extract entity type and ID from URI
                uri_str = str(obj)
                if uri_str.startswith(str(self.KC)):
                    parts = uri_str[len(str(self.KC)):].split('/', 1)
                    if len(parts) == 2:
                        ent_type, ent_id = parts
                        # Add to entities list
                        doc["entities"].append({
                            "type": ent_type.capitalize(),
                            "id": ent_id
                        })
            
            # Add to results
            results.append(doc)
        
        return results
    
    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query against the graph database.
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            List of result dictionaries
        """
        try:
            # Execute the query
            qres = self.graph.query(sparql_query)
            
            # Convert results to dictionaries
            results = []
            for row in qres:
                result = {}
                for i, var in enumerate(qres.vars):
                    result[var] = row[i]
                results.append(result)
            
            return results
        except Exception as e:
            print(f"SPARQL query error: {e}")
            return []
    
    def count_entities(self) -> int:
        """
        Count the number of entities in the graph.
        
        Returns:
            Number of entities
        """
        # Count all subjects with a type
        return len(set(self.graph.subjects(RDF.type, None)))
    
    def count_relationships(self) -> int:
        """
        Count the number of relationships in the graph.
        
        Returns:
            Number of relationships
        """
        # Count all triples, then subtract type statements
        total_triples = len(self.graph)
        type_statements = len(list(self.graph.triples((None, RDF.type, None))))
        
        # Also subtract statements with predicates in specific namespaces that are typically
        # used for properties rather than relationships
        property_predicates = [
            RDF.type,
            RDFS.label,
            RDFS.comment,
            DC.title,
            DC.description,
            DCTERMS.created,
            self.KC.uri
        ]
        
        property_statements = sum(len(list(self.graph.triples((None, pred, None)))) for pred in property_predicates)
        
        # Estimate the number of actual relationships
        return total_triples - property_statements

    def export_to_jsonld(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Export the graph to JSON-LD format.
        
        Args:
            output_path: Optional path to save the JSON-LD file
            
        Returns:
            Dictionary with the JSON-LD data and export info
        """
        # Serialize the graph to JSON-LD
        jsonld_data = self.graph.serialize(format="json-ld")
        
        if output_path:
            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(jsonld_data)
            
            return {
                "path": str(output_path),
                "triple_count": len(self.graph)
            }
        else:
            # Parse to dictionary
            import json
            jsonld_dict = json.loads(jsonld_data)
            
            return {
                "data": jsonld_dict,
                "triple_count": len(self.graph)
            }