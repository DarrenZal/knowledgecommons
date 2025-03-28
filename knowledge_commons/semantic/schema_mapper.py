"""
Schema Mapper Module
==================

Handles mapping between different schemas and formats in Knowledge Commons.
Converts between internal representations and standard formats like JSON-LD.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import rdflib
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF, SKOS, DC


class SchemaMapper:
    """Maps between different schema representations in Knowledge Commons."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the schema mapper.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.base_uri = config["schema"]["base_uri"]
        self.mappings_file = Path(config["schema"]["mappings_file"])
        
        # Load schema mappings
        self.mappings = self._load_mappings()
        
        # Initialize namespaces
        self.SCHEMA = Namespace("http://schema.org/")
        self.KC = Namespace(self.base_uri)
        
        # Default context for JSON-LD
        self.default_context = config["schema"].get("default_context", {})
        
    def _load_mappings(self) -> Dict[str, Any]:
        """
        Load schema mappings from configuration.
        
        Returns:
            Dictionary of schema mappings
        """
        if not self.mappings_file.exists():
            print(f"Warning: Schema mappings file not found: {self.mappings_file}")
            return {}
        
        try:
            with open(self.mappings_file, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading schema mappings: {e}")
            return {}
    
    def map_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map an entity to the internal schema.
        
        Args:
            entity: Source entity dictionary
            
        Returns:
            Mapped entity dictionary
        """
        entity_type = entity.get("type", "Thing")
        
        # Check if we have mappings for this entity type
        if entity_type not in self.mappings:
            # No specific mapping, return as is
            return entity
        
        type_mapping = self.mappings[entity_type]
        
        # Start with a copy of the original entity
        mapped_entity = entity.copy()
        
        # Apply property mappings if available
        if "properties" in type_mapping:
            property_mappings = type_mapping["properties"]
            
            for prop_name, prop_value in entity.items():
                if prop_name in ["id", "type"]:
                    # Keep ID and type as is
                    continue
                
                if prop_name in property_mappings:
                    # Property has a mapping
                    prop_mapping = property_mappings[prop_name]
                    
                    # Check for property type conversion
                    if "type" in prop_mapping:
                        if prop_mapping["type"] == "datetime" and isinstance(prop_value, str):
                            # Leave as string, but ensure it's ISO format
                            # (Conversion to proper datetime would happen when added to graph)
                            pass
                        elif prop_mapping["type"] == "list" and not isinstance(prop_value, list):
                            # Convert to list if it's not already
                            mapped_entity[prop_name] = [prop_value]
                        elif prop_mapping["type"] == "reference" and isinstance(prop_value, str):
                            # Convert string to reference object
                            if ":" in prop_value:
                                ref_type, ref_id = prop_value.split(":", 1)
                                mapped_entity[prop_name] = {
                                    "type": ref_type.capitalize(),
                                    "id": ref_id
                                }
                
                # No need to rename properties for internal representation
                
        return mapped_entity
    
    def to_jsonld(self, entity_type: str, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an entity to JSON-LD format.
        
        Args:
            entity_type: Type of entity
            entity: Entity dictionary
            
        Returns:
            JSON-LD representation
        """
        # Create the JSON-LD structure
        jsonld = {
            "@context": self._create_context(entity_type),
        }
        
        # Add ID if available
        if "id" in entity:
            jsonld["@id"] = f"{self.KC}{entity_type.lower()}/{entity['id']}"
        
        # Add type
        if entity_type == "Person":
            jsonld["@type"] = "Person"
        elif entity_type == "Organization":
            jsonld["@type"] = "Organization"
        elif entity_type == "Note":
            jsonld["@type"] = "CreativeWork"
        elif entity_type == "Event":
            jsonld["@type"] = "Event"
        elif entity_type == "Task":
            jsonld["@type"] = "Action"
        elif entity_type == "Project":
            jsonld["@type"] = "Project"
        elif entity_type == "Concept":
            jsonld["@type"] = "DefinedTerm"
        else:
            jsonld["@type"] = "Thing"
        
        # Check if we have mappings for this entity type
        type_mapping = self.mappings.get(entity_type, {})
        property_mappings = type_mapping.get("properties", {})
        
        # Convert properties based on mappings
        for prop_name, prop_value in entity.items():
            if prop_name in ["id", "type"]:
                # Already handled
                continue
            
            if prop_name in property_mappings and "schema_org" in property_mappings[prop_name]:
                # Use Schema.org property name if available
                json_key = property_mappings[prop_name]["schema_org"].split("/")[-1]
            else:
                # Use original property name
                json_key = prop_name
            
            # Handle different value types
            if prop_name in property_mappings and property_mappings[prop_name].get("type") == "reference":
                # Handle references to other entities
                if isinstance(prop_value, dict) and "id" in prop_value:
                    ref_type = prop_value.get("type", "Thing")
                    ref_id = prop_value["id"]
                    jsonld[json_key] = {
                        "@id": f"{self.KC}{ref_type.lower()}/{ref_id}"
                    }
                elif isinstance(prop_value, str) and ":" in prop_value:
                    # Format: "type:id"
                    ref_type, ref_id = prop_value.split(":", 1)
                    jsonld[json_key] = {
                        "@id": f"{self.KC}{ref_type.lower()}/{ref_id}"
                    }
            else:
                # Regular property
                jsonld[json_key] = prop_value
        
        return jsonld
    
    def from_jsonld(self, jsonld: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON-LD to internal entity representation.
        
        Args:
            jsonld: JSON-LD representation
            
        Returns:
            Entity dictionary
        """
        entity = {}
        
        # Extract type
        if "@type" in jsonld:
            json_type = jsonld["@type"]
            
            # Map JSON-LD type to internal type
            if json_type == "Person" or json_type == "http://schema.org/Person":
                entity["type"] = "Person"
            elif json_type == "Organization" or json_type == "http://schema.org/Organization":
                entity["type"] = "Organization"
            elif json_type == "CreativeWork" or json_type == "http://schema.org/CreativeWork":
                entity["type"] = "Note"
            elif json_type == "Event" or json_type == "http://schema.org/Event":
                entity["type"] = "Event"
            elif json_type == "Action" or json_type == "http://schema.org/Action":
                entity["type"] = "Task"
            elif json_type == "Project" or json_type == f"{self.KC}Project":
                entity["type"] = "Project"
            elif json_type == "DefinedTerm" or json_type == "http://schema.org/DefinedTerm":
                entity["type"] = "Concept"
            else:
                entity["type"] = "Thing"
        
        # Extract ID if available
        if "@id" in jsonld:
            id_str = jsonld["@id"]
            
            # Extract ID from URI
            if "/" in id_str:
                entity["id"] = id_str.split("/")[-1]
        
        # Get entity type for property mapping
        entity_type = entity.get("type", "Thing")
        type_mapping = self.mappings.get(entity_type, {})
        property_mappings = type_mapping.get("properties", {})
        
        # Create reverse mapping from Schema.org to internal properties
        schema_to_internal = {}
        for prop_name, prop_mapping in property_mappings.items():
            if "schema_org" in prop_mapping:
                schema_prop = prop_mapping["schema_org"].split("/")[-1]
                schema_to_internal[schema_prop] = prop_name
        
        # Process other properties
        for json_key, json_value in jsonld.items():
            if json_key.startswith("@"):
                # Skip JSON-LD keywords
                continue
            
            # Map Schema.org property to internal property
            if json_key in schema_to_internal:
                prop_name = schema_to_internal[json_key]
            else:
                prop_name = json_key
            
            # Process property value
            if isinstance(json_value, dict) and "@id" in json_value:
                # Reference to another entity
                ref_id = json_value["@id"].split("/")[-1]
                ref_type = "Thing"
                
                # Try to determine reference type from URI
                uri_parts = json_value["@id"].split("/")
                if len(uri_parts) >= 2:
                    possible_type = uri_parts[-2]
                    ref_type = possible_type.capitalize()
                
                entity[prop_name] = {
                    "type": ref_type,
                    "id": ref_id
                }
            else:
                # Regular property
                entity[prop_name] = json_value
        
        return entity
    
    def to_rdf(self, entity_type: str, entity: Dict[str, Any]) -> Graph:
        """
        Convert an entity to RDF graph.
        
        Args:
            entity_type: Type of entity
            entity: Entity dictionary
            
        Returns:
            RDF graph containing the entity
        """
        graph = Graph()
        
        # Bind namespaces
        graph.bind("schema", self.SCHEMA)
        graph.bind("kc", self.KC)
        graph.bind("rdf", RDF)
        graph.bind("rdfs", RDFS)
        graph.bind("xsd", XSD)
        graph.bind("foaf", FOAF)
        graph.bind("skos", SKOS)
        graph.bind("dc", DC)
        
        # Create entity URI
        entity_id = entity.get("id", str(hash(str(entity))))
        entity_uri = URIRef(f"{self.KC}{entity_type.lower()}/{entity_id}")
        
        # Add type triple based on entity type
        if entity_type == "Person":
            graph.add((entity_uri, RDF.type, FOAF.Person))
        elif entity_type == "Organization":
            graph.add((entity_uri, RDF.type, FOAF.Organization))
        elif entity_type == "Note":
            graph.add((entity_uri, RDF.type, self.KC.Note))
        elif entity_type == "Event":
            graph.add((entity_uri, RDF.type, self.SCHEMA.Event))
        elif entity_type == "Task":
            graph.add((entity_uri, RDF.type, self.KC.Task))
        elif entity_type == "Project":
            graph.add((entity_uri, RDF.type, self.KC.Project))
        elif entity_type == "Concept":
            graph.add((entity_uri, RDF.type, SKOS.Concept))
        else:
            graph.add((entity_uri, RDF.type, self.SCHEMA.Thing))
        
        # Get property mappings for this entity type
        type_mapping = self.mappings.get(entity_type, {})
        property_mappings = type_mapping.get("properties", {})
        
        # Add properties based on mappings
        for prop_name, prop_value in entity.items():
            if prop_name in ["id", "type"]:
                # Already handled
                continue
            
            # Determine predicate based on mapping
            if prop_name in property_mappings and "rdf" in property_mappings[prop_name]:
                predicate = URIRef(property_mappings[prop_name]["rdf"])
            else:
                # Use KC namespace for unmapped properties
                predicate = self.KC[prop_name]
            
            # Handle different value types
            if prop_name in property_mappings:
                prop_type = property_mappings[prop_name].get("type")
                
                if prop_type == "datetime":
                    # Add with XSD datetime type
                    graph.add((entity_uri, predicate, Literal(prop_value, datatype=XSD.dateTime)))
                elif prop_type == "list":
                    # Add multiple triples for list values
                    if isinstance(prop_value, list):
                        for item in prop_value:
                            graph.add((entity_uri, predicate, Literal(item)))
                    else:
                        # Single value
                        graph.add((entity_uri, predicate, Literal(prop_value)))
                elif prop_type == "reference":
                    # Reference to another entity
                    if isinstance(prop_value, dict) and "id" in prop_value:
                        ref_type = prop_value.get("type", "Thing").lower()
                        ref_id = prop_value["id"]
                        ref_uri = URIRef(f"{self.KC}{ref_type}/{ref_id}")
                        graph.add((entity_uri, predicate, ref_uri))
                    elif isinstance(prop_value, str) and ":" in prop_value:
                        # Format: "type:id"
                        ref_type, ref_id = prop_value.split(":", 1)
                        ref_uri = URIRef(f"{self.KC}{ref_type.lower()}/{ref_id}")
                        graph.add((entity_uri, predicate, ref_uri))
                    else:
                        # Fallback: treat as literal
                        graph.add((entity_uri, predicate, Literal(prop_value)))
                else:
                    # Default: add as literal
                    graph.add((entity_uri, predicate, Literal(prop_value)))
            else:
                # No mapping, add as literal
                graph.add((entity_uri, predicate, Literal(prop_value)))
        
        return graph
    
    def from_rdf(self, graph: Graph, entity_uri: URIRef) -> Dict[str, Any]:
        """
        Convert RDF representation to internal entity.
        
        Args:
            graph: RDF graph containing the entity
            entity_uri: URI of the entity
            
        Returns:
            Entity dictionary
        """
        entity = {}
        
        # Extract ID from URI
        uri_str = str(entity_uri)
        if "/" in uri_str:
            entity["id"] = uri_str.split("/")[-1]
        
        # Determine entity type from RDF type
        entity_type = "Thing"  # Default
        
        type_triples = list(graph.triples((entity_uri, RDF.type, None)))
        for _, _, rdf_type in type_triples:
            if rdf_type == FOAF.Person:
                entity_type = "Person"
                break
            elif rdf_type == FOAF.Organization:
                entity_type = "Organization"
                break
            elif rdf_type == self.KC.Note:
                entity_type = "Note"
                break
            elif rdf_type == self.SCHEMA.Event:
                entity_type = "Event"
                break
            elif rdf_type == self.KC.Task:
                entity_type = "Task"
                break
            elif rdf_type == self.KC.Project:
                entity_type = "Project"
                break
            elif rdf_type == SKOS.Concept:
                entity_type = "Concept"
                break
            elif rdf_type == self.SCHEMA.Thing:
                entity_type = "Thing"
                break
        
        entity["type"] = entity_type
        
        # Get property mappings for this entity type
        type_mapping = self.mappings.get(entity_type, {})
        property_mappings = type_mapping.get("properties", {})
        
        # Create reverse mapping from RDF predicates to property names
        rdf_to_property = {}
        for prop_name, prop_mapping in property_mappings.items():
            if "rdf" in prop_mapping:
                rdf_predicate = prop_mapping["rdf"]
                rdf_to_property[rdf_predicate] = prop_name
        
        # Process all predicates for this entity
        for s, p, o in graph.triples((entity_uri, None, None)):
            # Skip RDF type triples (already processed)
            if p == RDF.type:
                continue
            
            # Determine property name
            predicate_uri = str(p)
            
            if predicate_uri in rdf_to_property:
                # Use mapped property name
                prop_name = rdf_to_property[predicate_uri]
            elif predicate_uri.startswith(str(self.KC)):
                # KC namespace property
                prop_name = predicate_uri[len(str(self.KC)):]
            else:
                # Use full predicate URI as property name
                prop_name = predicate_uri
            
            # Process object based on type
            if isinstance(o, URIRef):
                # Reference to another entity
                obj_uri = str(o)
                
                # Extract type and ID from URI
                if obj_uri.startswith(str(self.KC)) and "/" in obj_uri:
                    parts = obj_uri[len(str(self.KC)):].split("/", 1)
                    if len(parts) == 2:
                        ref_type, ref_id = parts
                        
                        # Check if property is mapped as a reference
                        is_reference = False
                        if prop_name in property_mappings:
                            is_reference = property_mappings[prop_name].get("type") == "reference"
                        
                        if is_reference:
                            # Add as reference object
                            entity[prop_name] = {
                                "type": ref_type.capitalize(),
                                "id": ref_id
                            }
                        else:
                            # Add as string reference
                            entity[prop_name] = f"{ref_type}:{ref_id}"
                    else:
                        # Can't parse reference, use URI as is
                        entity[prop_name] = obj_uri
                else:
                    # External URI
                    entity[prop_name] = obj_uri
            elif isinstance(o, Literal):
                # Literal value
                value = o.toPython()
                
                # Check if property is mapped as a list
                if prop_name in property_mappings and property_mappings[prop_name].get("type") == "list":
                    # Add to list if property already exists
                    if prop_name in entity:
                        if not isinstance(entity[prop_name], list):
                            entity[prop_name] = [entity[prop_name]]
                        entity[prop_name].append(value)
                    else:
                        entity[prop_name] = [value]
                else:
                    # Regular property
                    entity[prop_name] = value
            else:
                # Other (e.g., BNode)
                entity[prop_name] = str(o)
        
        return entity
    
    def _create_context(self, entity_type: str) -> Dict[str, Any]:
        """
        Create JSON-LD context for an entity type.
        
        Args:
            entity_type: Type of entity
            
        Returns:
            JSON-LD context object
        """
        # Start with default context
        context = dict(self.default_context)
        
        # Add entity-specific mappings
        type_mapping = self.mappings.get(entity_type, {})
        property_mappings = type_mapping.get("properties", {})
        
        for prop_name, prop_mapping in property_mappings.items():
            if "schema_org" in prop_mapping:
                # Map property to Schema.org
                schema_prop = prop_mapping["schema_org"].split("/")[-1]
                context[prop_name] = f"schema:{schema_prop}"
        
        return context
    
    def map_to_schema(self, schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map data to a specific schema.
        
        Args:
            schema_name: Name of target schema
            data: Data to map
            
        Returns:
            Mapped data
        """
        if schema_name == "jsonld":
            # Convert to JSON-LD
            entity_type = data.get("type", "Thing")
            return self.to_jsonld(entity_type, data)
        elif schema_name == "rdf":
            # Convert to RDF (return serialized Turtle)
            entity_type = data.get("type", "Thing")
            graph = self.to_rdf(entity_type, data)
            return graph.serialize(format="turtle")
        else:
            # Check if we have explicit mappings for this schema
            if schema_name not in self.mappings:
                # No mapping available
                return data
            
            # Get schema mapping
            schema_mapping = self.mappings[schema_name]
            
            # Apply mapping to data
            # (This is a simplified version - a real implementation would be more complex)
            mapped_data = {}
            
            for key, value in data.items():
                if key in schema_mapping:
                    # Use mapped key
                    mapped_key = schema_mapping[key]
                    mapped_data[mapped_key] = value
                else:
                    # Keep original key
                    mapped_data[key] = value
            
            return mapped_data