"""
Vector Database Module
=====================

Handles interaction with vector databases for Knowledge Commons.
Supports Chroma and Qdrant for embedding storage and retrieval.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Import vector database libraries conditionally
# to avoid hard dependencies
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class VectorDatabase:
    """
    Manages vector database interactions for Knowledge Commons.
    Supports multiple vector database backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector database manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.db_type = config["databases"]["vector"]["type"]
        self.db_path = Path(config["databases"]["vector"]["path"])
        self.db_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize the vector database
        self.db = self._init_db()
        
        # Collection/index name
        self.collection_name = "knowledge_commons"
        
        # Vector dimensions
        self.dimensions = config["databases"]["vector"].get("dimensions", 1536)
    
    def _init_db(self) -> Any:
        """
        Initialize the vector database based on configuration.
        
        Returns:
            Initialized vector database client
        """
        if self.db_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ImportError("ChromaDB is not installed. Install with 'pip install chromadb'")
            
            # Create a persistent ChromaDB client
            return chromadb.PersistentClient(path=str(self.db_path))
        
        elif self.db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant client is not installed. Install with 'pip install qdrant-client'")
            
            # Initialize Qdrant client
            if self.config["databases"]["vector"]["remote"]["enabled"]:
                # Use remote Qdrant server
                url = self.config["databases"]["vector"]["remote"]["url"]
                api_key = self.config["databases"]["vector"]["remote"].get("api_key")
                
                return QdrantClient(url=url, api_key=api_key)
            else:
                # Use local Qdrant
                return QdrantClient(path=str(self.db_path / "qdrant_data"))
        
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    def _ensure_collection(self):
        """Ensure the collection/index exists."""
        if self.db_type == "chroma":
            # Get or create collection
            try:
                self.collection = self.db.get_collection(name=self.collection_name)
                print(f"Using existing collection: {self.collection_name}")
            except Exception as e:
                # Collection doesn't exist, create it
                print(f"Creating new collection: {self.collection_name}")
                self.collection = self.db.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
        
        elif self.db_type == "qdrant":
            # Check if collection exists
            collections = self.db.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create the collection
                self.db.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.dimensions,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
    
    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """
        Add an embedding to the vector database.
        
        Args:
            id: Unique identifier for the embedding
            embedding: Vector embedding (list of floats)
            metadata: Metadata to store with the embedding
            
        Returns:
            String identifier for the embedding
        """
        self._ensure_collection()
        
        # Fix for ChromaDB: Convert lists to strings in metadata
        if self.db_type == "chroma":
            # Create a copy of metadata to avoid modifying the original
            processed_metadata = {}
            for key, value in metadata.items():
                # Convert lists to strings
                if isinstance(value, list):
                    processed_metadata[key] = ", ".join(str(item) for item in value)
                else:
                    processed_metadata[key] = value
        else:
            processed_metadata = metadata
        
        if self.db_type == "chroma":
            # Add to ChromaDB with processed metadata
            self.collection.upsert(
                ids=[id],
                embeddings=[embedding],
                metadatas=[processed_metadata]
            )
        
        elif self.db_type == "qdrant":
            # Add to Qdrant
            self.db.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=id,
                        vector=embedding,
                        payload=metadata  # Qdrant accepts more complex data types
                    )
                ]
            )
        
        return id
    
    def search(self, 
               query_embedding: List[float], 
               limit: int = 5, 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            filters: Optional filters to apply to results
            
        Returns:
            List of results with scores and metadata
        """
        self._ensure_collection()
        results = []
        
        if self.db_type == "chroma":
            # Convert filters to ChromaDB format
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = value
            
            # Search in ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause
            )
            
            # Format results
            for i in range(len(search_results["ids"][0])):
                result_id = search_results["ids"][0][i]
                metadata = search_results["metadatas"][0][i]
                distance = search_results["distances"][0][i]
                
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - distance
                
                results.append({
                    "id": result_id,
                    "score": score,
                    "metadata": metadata
                })
        
        elif self.db_type == "qdrant":
            # Convert filters to Qdrant format
            filter_query = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # For lists, check if any item matches
                        or_conditions = [
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchValue(value=v)
                            )
                            for v in value
                        ]
                        filter_conditions.append(qdrant_models.Filter(
                            should=or_conditions,
                            min_should=1
                        ))
                    else:
                        # For single values, exact match
                        filter_conditions.append(qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value)
                        ))
                
                filter_query = qdrant_models.Filter(
                    must=filter_conditions
                )
            
            # Search in Qdrant
            search_results = self.db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_query
            )
            
            # Format results
            for scored_point in search_results:
                results.append({
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "metadata": scored_point.payload
                })
        
        return results
    
    def get_embedding(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get an embedding by ID.
        
        Args:
            id: Embedding identifier
            
        Returns:
            Dictionary with embedding and metadata, or None if not found
        """
        self._ensure_collection()
        
        if self.db_type == "chroma":
            try:
                # Get from ChromaDB
                result = self.collection.get(
                    ids=[id],
                    include=["embeddings", "metadatas"]
                )
                
                if result["ids"] and result["ids"][0] == id:
                    return {
                        "id": id,
                        "embedding": result["embeddings"][0],
                        "metadata": result["metadatas"][0]
                    }
                return None
            except Exception:
                return None
        
        elif self.db_type == "qdrant":
            try:
                # Get from Qdrant
                result = self.db.retrieve(
                    collection_name=self.collection_name,
                    ids=[id],
                    with_vectors=True,
                    with_payload=True
                )
                
                if result:
                    point = result[0]
                    return {
                        "id": point.id,
                        "embedding": point.vector,
                        "metadata": point.payload
                    }
                return None
            except Exception:
                return None
    
    def delete_embedding(self, id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            id: Embedding identifier
            
        Returns:
            True if successful, False otherwise
        """
        self._ensure_collection()
        
        try:
            if self.db_type == "chroma":
                # Delete from ChromaDB
                self.collection.delete(ids=[id])
            
            elif self.db_type == "qdrant":
                # Delete from Qdrant
                self.db.delete(
                    collection_name=self.collection_name,
                    points_selector=qdrant_models.PointIdsList(points=[id])
                )
                
            return True
        except Exception:
            return False
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an embedding.
        
        Args:
            id: Embedding identifier
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        self._ensure_collection()
        
        try:
            if self.db_type == "chroma":
                # Get existing embedding
                result = self.collection.get(
                    ids=[id],
                    include=["embeddings"]
                )
                
                if not result["ids"]:
                    return False
                
                # Update with new metadata
                self.collection.upsert(
                    ids=[id],
                    embeddings=[result["embeddings"][0]],
                    metadatas=[metadata]
                )
            
            elif self.db_type == "qdrant":
                # Update metadata in Qdrant
                self.db.set_payload(
                    collection_name=self.collection_name,
                    payload=metadata,
                    points=[id]
                )
                
            return True
        except Exception as e:
            print(f"Error updating metadata: {e}")
            return False
    
    def count_vectors(self) -> int:
        """
        Count the number of vectors in the database.
        
        Returns:
            Number of vectors
        """
        try:
            self._ensure_collection()
            
            if self.db_type == "chroma":
                # Get count from ChromaDB
                return self.collection.count()
            
            elif self.db_type == "qdrant":
                # Get count from Qdrant
                collection_info = self.db.get_collection(collection_name=self.collection_name)
                return collection_info.vectors_count
                
            return 0
        except Exception as e:
            print(f"Error counting vectors: {e}")
            return 0
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector database.
        
        Returns:
            List of collection names
        """
        if self.db_type == "chroma":
            return [c.name for c in self.db.list_collections()]
        
        elif self.db_type == "qdrant":
            collections = self.db.get_collections().collections
            return [c.name for c in collections]
        
        return []
    
    def similarity_search(self, 
                         text_embedding: List[float], 
                         limit: int = 5, 
                         filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search with text embedding.
        
        Args:
            text_embedding: Text embedding vector
            limit: Maximum number of results
            filters: Optional filters to apply
            
        Returns:
            List of search results
        """
        # This is just a wrapper around search with a more descriptive name
        return self.search(
            query_embedding=text_embedding,
            limit=limit,
            filters=filters
        )
    
    def bulk_add_embeddings(self, 
                           embeddings: List[List[float]], 
                           ids: List[str], 
                           metadatas: List[Dict[str, Any]]) -> bool:
        """
        Add multiple embeddings at once.
        
        Args:
            embeddings: List of embedding vectors
            ids: List of corresponding IDs
            metadatas: List of corresponding metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if len(embeddings) != len(ids) or len(embeddings) != len(metadatas):
            raise ValueError("Embeddings, IDs, and metadatas must have the same length")
        
        self._ensure_collection()
        
        try:
            if self.db_type == "chroma":
                # Bulk add to ChromaDB
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            
            elif self.db_type == "qdrant":
                # Bulk add to Qdrant
                points = [
                    qdrant_models.PointStruct(
                        id=id,
                        vector=embedding,
                        payload=metadata
                    )
                    for id, embedding, metadata in zip(ids, embeddings, metadatas)
                ]
                
                self.db.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
            return True
        except Exception as e:
            print(f"Error adding embeddings: {e}")
            return False