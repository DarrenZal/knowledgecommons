"""
Query Engine Module
=================

Provides the core querying capabilities for Knowledge Commons.
Combines vector search and graph queries for comprehensive retrieval.
"""

import re
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from knowledge_commons.core.graph_db import GraphDatabase
from knowledge_commons.core.vector_db import VectorDatabase
from knowledge_commons.core.local_storage import LocalStorage
from knowledge_commons.semantic.extraction import EntityExtractor


class QueryEngine:
    """
    Handles knowledge retrieval across multiple data stores.
    Combines vector similarity search with graph-based queries.
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                graph_db: GraphDatabase,
                vector_db: VectorDatabase,
                storage: LocalStorage):
        """
        Initialize the query engine.
        
        Args:
            config: System configuration
            graph_db: Graph database instance
            vector_db: Vector database instance
            storage: Local storage instance
        """
        self.config = config
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.storage = storage
        
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(config)
        
        # Default search settings
        self.default_limit = config["query_engine"].get("default_search_limit", 10)
        self.hybrid_search = config["query_engine"].get("hybrid_search", {}).get("enabled", True)
        self.vector_weight = config["query_engine"].get("hybrid_search", {}).get("vector_weight", 0.7)
        self.graph_weight = config["query_engine"].get("hybrid_search", {}).get("graph_weight", 0.3)
    
    def query(self, 
             query_text: str, 
             query_type: Optional[str] = None,
             limit: int = None) -> Dict[str, Any]:
        """
        Query the knowledge system.
        
        Args:
            query_text: Text of the query
            query_type: Optional type of query ('vector', 'graph', 'hybrid')
            limit: Maximum number of results
            
        Returns:
            Dictionary containing query results
        """
        # Use default limit if not specified
        if limit is None:
            limit = self.default_limit
        
        # Determine query type if not specified
        if query_type is None:
            # Auto-detect based on query structure
            if self._is_structured_query(query_text):
                query_type = "graph"
            else:
                query_type = "hybrid" if self.hybrid_search else "vector"
        
        # Extract entities from the query
        entities = self.entity_extractor.extract(query_text)
        
        # Process the query based on its type
        if query_type == "vector":
            results = self._vector_search(query_text, limit)
        elif query_type == "graph":
            results = self._graph_search(query_text, entities, limit)
        elif query_type == "hybrid":
            results = self._hybrid_search(query_text, entities, limit)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Process results and add content from local storage if needed
        return self._process_results(results, query_text)
    
    def _is_structured_query(self, query_text: str) -> bool:
        """
        Determine if the query appears to be structured.
        
        Args:
            query_text: Text of the query
            
        Returns:
            True if the query appears structured, False otherwise
        """
        # Look for specific structured query patterns
        structured_patterns = [
            r"\bfind\s+(all|the)\s+(.+?)\s+where\b",
            r"\bshow\s+(all|me)\s+(.+?)\s+with\b",
            r"\bwhat\s+(.+?)\s+has\s+(.+?)\b",
            r"\bwho\s+(.+?)\s+with\s+(.+?)\b",
            r"\brelationship\s+between\b"
        ]
        
        for pattern in structured_patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
                return True
        
        # Check if the query mentions entity types from our schema
        entity_types = ["person", "organization", "project", "task", "event", "note", "concept"]
        for entity_type in entity_types:
            pattern = r"\b" + entity_type + r"s?\b"
            if re.search(pattern, query_text, re.IGNORECASE):
                return True
        
        return False
    
    def _vector_search(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_text: Text of the query
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        # Generate embedding for the query
        query_embedding = self.entity_extractor.generate_embedding(query_text)
        
        if query_embedding is None:
            return []
        
        # Perform vector search
        search_results = self.vector_db.search(
            query_embedding=query_embedding,
            limit=limit
        )
        
        # Format results
        results = []
        for item in search_results:
            result = {
                "id": item["id"],
                "score": item["score"],
                "type": item["metadata"].get("type", "unknown"),
                "title": item["metadata"].get("title", "Untitled"),
                "path": item["metadata"].get("path", ""),
                "source": "vector"
            }
            
            results.append(result)
        
        return results
    
    def _graph_search(self, query_text: str, entities: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        Perform graph-based search.
        
        Args:
            query_text: Text of the query
            entities: Extracted entities from the query
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        # Find documents related to the entities in the query
        if entities:
            graph_results = self.graph_db.find_related_documents(entities, limit=limit)
        else:
            # If no entities were extracted, fall back to basic search
            # Find documents with matching properties
            properties = {"title": query_text}  # Simple text match in title
            graph_results = self.graph_db.search_entities("Note", properties, limit=limit)
        
        # Format results
        results = []
        for item in graph_results:
            # Skip items without a path
            if "path" not in item:
                continue
                
            result = {
                "id": item.get("id", str(uuid.uuid4())),
                "score": 1.0,  # Graph results don't have scores by default
                "type": item.get("type", "note"),
                "title": item.get("title", "Untitled"),
                "path": item.get("path", ""),
                "source": "graph",
                "entities": item.get("entities", [])
            }
            
            results.append(result)
        
        return results
    
    def _hybrid_search(self, 
                      query_text: str, 
                      entities: List[Dict[str, Any]], 
                      limit: int) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and graph approaches.
        
        Args:
            query_text: Text of the query
            entities: Extracted entities from the query
            limit: Maximum number of results
            
        Returns:
            List of result dictionaries
        """
        # Get results from both sources
        vector_results = self._vector_search(query_text, limit * 2)  # Get more to allow for merging
        graph_results = self._graph_search(query_text, entities, limit * 2)
        
        # Combine results
        combined_results = self._combine_results(vector_results, graph_results, limit)
        
        return combined_results
    
    def _combine_results(self, 
                        vector_results: List[Dict[str, Any]], 
                        graph_results: List[Dict[str, Any]], 
                        limit: int) -> List[Dict[str, Any]]:
        """
        Combine and rank results from different sources.
        
        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            limit: Maximum number of results to return
            
        Returns:
            Combined and ranked list of results
        """
        # Create a dictionary to track combined scores by document path
        combined_scores = {}
        
        # Process vector results
        for result in vector_results:
            path = result["path"]
            vector_score = result["score"] * self.vector_weight
            
            combined_scores[path] = {
                "id": result["id"],
                "title": result["title"],
                "type": result["type"],
                "path": path,
                "vector_score": vector_score,
                "graph_score": 0.0,
                "total_score": vector_score,
                "sources": ["vector"]
            }
        
        # Process graph results
        for result in graph_results:
            path = result["path"]
            graph_score = result["score"] * self.graph_weight
            
            if path in combined_scores:
                # Update existing entry
                combined_scores[path]["graph_score"] = graph_score
                combined_scores[path]["total_score"] += graph_score
                combined_scores[path]["sources"].append("graph")
                
                # Add entities if available
                if "entities" in result and result["entities"]:
                    combined_scores[path]["entities"] = result["entities"]
            else:
                # Create new entry
                combined_scores[path] = {
                    "id": result["id"],
                    "title": result["title"],
                    "type": result["type"],
                    "path": path,
                    "vector_score": 0.0,
                    "graph_score": graph_score,
                    "total_score": graph_score,
                    "sources": ["graph"]
                }
                
                # Add entities if available
                if "entities" in result and result["entities"]:
                    combined_scores[path]["entities"] = result["entities"]
        
        # Convert to list and sort by total score
        results = list(combined_scores.values())
        results.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Take top results up to limit
        return results[:limit]
    
    def _process_results(self, results: List[Dict[str, Any]], query_text: str) -> Dict[str, Any]:
        """
        Process search results to add content snippets and other information.
        
        Args:
            results: Search results
            query_text: Original query text
            
        Returns:
            Dictionary with processed results
        """
        processed_items = []
        
        for result in results:
            # Get path to document
            path = result.get("path")
            if not path:
                continue
            
            path_obj = Path(path)
            if not path_obj.exists():
                continue
            
            # Load document content based on type
            content = None
            if path_obj.suffix.lower() == ".md":
                try:
                    doc_data = self.storage.load_markdown(path)
                    content = doc_data["content"]
                    metadata = doc_data["metadata"]
                    
                    # Update title if it wasn't set
                    if result["title"] == "Untitled" and "title" in metadata:
                        result["title"] = metadata["title"]
                except Exception as e:
                    print(f"Error loading markdown: {e}")
            elif path_obj.suffix.lower() == ".jsonld":
                try:
                    doc_data = self.storage.load_jsonld(path)
                    if "data" in doc_data and isinstance(doc_data["data"], dict):
                        # Extract text content if available
                        if "content" in doc_data["data"]:
                            content = doc_data["data"]["content"]
                        # Or try schema.org text field
                        elif "http://schema.org/text" in doc_data["data"]:
                            content = doc_data["data"]["http://schema.org/text"]
                except Exception as e:
                    print(f"Error loading JSON-LD: {e}")
            
            # Generate a snippet from the content
            snippet = None
            if content:
                snippet = self._generate_snippet(content, query_text)
            
            # Create processed result
            processed_result = {
                "id": result["id"],
                "title": result["title"],
                "type": result["type"],
                "path": path,
                "score": result.get("total_score", result.get("score", 0.0)),
                "snippet": snippet
            }
            
            # Add entities if available
            if "entities" in result:
                processed_result["entities"] = result["entities"]
            
            processed_items.append(processed_result)
        
        # Return the final results object
        return {
            "query": query_text,
            "count": len(processed_items),
            "items": processed_items
        }
    
    def _generate_snippet(self, content: str, query_text: str, max_length: int = 200) -> str:
        """Generate a relevant snippet from content based on query."""
        # Handle newlines in content for better matching
        content_for_matching = content.replace('\n', ' ')
        query_terms = set(query_text.lower().split())
        
        # Split content by paragraphs, preserving newlines for display
        paragraphs = content.split("\n\n")
        
        best_paragraph = None
        best_score = 0
        
        for paragraph in paragraphs:
            # Skip very short paragraphs or headings
            if len(paragraph) < 10 or paragraph.strip().startswith("#"):
                continue
                
            # Prepare paragraph for matching (remove newlines for comparison only)
            paragraph_for_matching = paragraph.replace('\n', ' ').lower()
            
            # Count matching terms
            score = sum(1 for term in query_terms if term in paragraph_for_matching)
            
            # Check for section headers that might match
            if "action item" in paragraph_for_matching.lower() or "task" in paragraph_for_matching.lower():
                score += 3  # Give extra weight to sections about tasks/actions
            
            if score > best_score:
                best_score = score
                best_paragraph = paragraph
        
        # If no good paragraph found, just use the beginning of the document
        if best_paragraph is None:
            # Remove markdown headings
            content_no_headings = re.sub(r'^#.*$', '', content, flags=re.MULTILINE)
            # Get the first non-empty paragraph
            for paragraph in content_no_headings.split("\n\n"):
                if paragraph.strip():
                    best_paragraph = paragraph
                    break
            
            if best_paragraph is None:
                return None
        
        # Truncate to max length if needed, preserving newlines
        if len(best_paragraph) > max_length:
            # Try to truncate at word boundary
            truncated = best_paragraph[:max_length]
            last_space = truncated.rfind(" ")
            if last_space > max_length * 0.8:  # Only adjust if we're not losing too much
                truncated = truncated[:last_space]
            return truncated + "..."
        
        return best_paragraph