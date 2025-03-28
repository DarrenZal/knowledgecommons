"""
Entity Extraction Module
======================

Provides entity extraction and concept recognition for Knowledge Commons.
Also handles embedding generation for text.
"""

import re
import sys
import uuid
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import importlib.util
import os
from pathlib import Path
import requests

# Check for optional dependencies
ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# Display available modules for debugging
print(f"Anthropic available: {ANTHROPIC_AVAILABLE}")
print(f"Sentence Transformers available: {SENTENCE_TRANSFORMERS_AVAILABLE}")
print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
print(f"PyTorch available: {TORCH_AVAILABLE}")


class EntityExtractor:
    """
    Extracts entities and concepts from text.
    Also generates embeddings for semantic search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the entity extractor.
        
        Args:
            config: System configuration
        """
        self.config = config
        self._init_llm()
        self._init_embedding_model()
    
    def _init_llm(self):
        """Initialize the LLM client based on configuration."""
        self.llm_enabled = self.config["llm"]["enabled"]
        self.llm_provider = self.config["llm"]["provider"]
        
        if not self.llm_enabled:
            self.llm_client = None
            return
        
        if self.llm_provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                print("Warning: Anthropic package not installed. Installing...")
                try:
                    import pip
                    pip.main(['install', 'anthropic'])
                    import anthropic
                    
                    # Get API key from environment or config
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        # Try to get from config
                        api_key = self.config["llm"].get("api_key")
                    
                    if not api_key:
                        print("Error: ANTHROPIC_API_KEY not found in environment or config")
                        self.llm_client = None
                        return
                    
                    print(f"Using API key starting with: {api_key[:10]}...")
                    self.llm_client = anthropic.Anthropic(api_key=api_key)
                except Exception as e:
                    print(f"Error installing anthropic: {e}")
                    self.llm_client = None
            else:
                import anthropic
                
                # Get API key from environment or config
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    # Try to get from config
                    api_key = self.config["llm"].get("api_key")
                
                if not api_key:
                    print("Error: ANTHROPIC_API_KEY not found in environment or config")
                    self.llm_client = None
                    return
                
                print(f"Using API key starting with: {api_key[:10]}...")
                self.llm_client = anthropic.Anthropic(api_key=api_key)
        
        elif self.llm_provider == "openai":
            try:
                import openai
                self.llm_client = openai.OpenAI(
                    api_key=self.config["llm"].get("api_key", os.environ.get("OPENAI_API_KEY"))
                )
            except ImportError:
                print("Warning: OpenAI package not installed. Entity extraction may be limited.")
                self.llm_client = None
        
        else:
            print(f"Warning: Unsupported LLM provider: {self.llm_provider}")
            self.llm_client = None
    
    def _init_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        embeddings_config = self.config["llm"].get("embeddings", {})
        embedding_provider = embeddings_config.get("provider", "huggingface")
        embedding_model = embeddings_config.get("model", "Snowflake/snowflake-arctic-embed-xs")
        
        self.embedding_model = None
        self.hf_model = None
        self.tokenizer = None
        
        # Initialize based on provider
        if embedding_provider == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                print("Transformers package not installed. Installing...")
                try:
                    import pip
                    pip.main(['install', 'transformers', 'torch'])
                    # Don't try to update global variables here - it won't work properly
                    # Just continue with the import attempts
                except Exception as e:
                    print(f"Error installing transformers: {e}")
                    return
            
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                print(f"Loading Hugging Face model: {embedding_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
                self.hf_model = AutoModel.from_pretrained(embedding_model)
                self.embedding_model = "huggingface_loaded"
                print("Successfully loaded Hugging Face model")
            except Exception as e:
                print(f"Error loading Hugging Face model: {e}")
                
                # Fall back to sentence-transformers
                self._try_sentence_transformers(embedding_model)
                
        elif embedding_provider == "sentence_transformers":
            self._try_sentence_transformers(embedding_model)
                
    def _try_sentence_transformers(self, model_name):
        """Try to load a model using sentence-transformers as fallback."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("sentence-transformers not installed. Installing...")
            try:
                import pip
                pip.main(['install', 'sentence-transformers'])
                # Don't try to modify global variables after definition
            except Exception as e:
                print(f"Error installing sentence-transformers: {e}")
                return
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence-transformers model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            print("Successfully loaded sentence-transformers model")
        except Exception as e:
            print(f"Error loading sentence-transformers model: {e}")
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entity dictionaries
        """
        if not self.llm_enabled or not self.llm_client:
            # Fall back to rule-based extraction
            return self._rule_based_extraction(text)
        
        try:
            # Use LLM for entity extraction
            if self.llm_provider == "anthropic":
                return self._extract_with_anthropic(text)
            elif self.llm_provider == "openai":
                return self._extract_with_openai(text)
            else:
                return self._rule_based_extraction(text)
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return self._rule_based_extraction(text)
    
    def _extract_with_anthropic(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using Anthropic's Claude.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entity dictionaries
        """
        prompt = f"""
        Extract the key entities from the following text. Focus on people, organizations, 
        places, projects, concepts, and events. For each entity, provide its type and 
        any additional information available in the text.
        
        Return the results as a JSON array where each object has these properties:
        - id: A unique identifier (UUID)
        - type: The entity type (Person, Organization, Place, Project, Concept, Event)
        - name: The entity name
        - description: Brief description or context (if available in the text)
        - Any other relevant properties for the entity type
        
        Only include entities that are clearly mentioned in the text, with no speculations.
        Only return the JSON array with no additional text.
        
        Text to analyze:
        --------------
        {text}
        """
        
        try:
            response = self.llm_client.messages.create(
                model=self.config["llm"]["model"],
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'(\[.*\])', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                entities = json.loads(json_str)
                
                # Normalize entity types
                for entity in entities:
                    if "type" in entity:
                        entity["type"] = entity["type"].capitalize()
                
                return entities
            else:
                print("Failed to extract JSON from LLM response")
                return []
                
        except Exception as e:
            print(f"Error in Anthropic entity extraction: {e}")
            return []
    
    def _extract_with_openai(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using OpenAI's API.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entity dictionaries
        """
        prompt = f"""
        Extract the key entities from the following text. Focus on people, organizations, 
        places, projects, concepts, and events. For each entity, provide its type and 
        any additional information available in the text.
        
        Return the results as a JSON array where each object has these properties:
        - id: A unique identifier (UUID)
        - type: The entity type (Person, Organization, Place, Project, Concept, Event)
        - name: The entity name
        - description: Brief description or context (if available in the text)
        - Any other relevant properties for the entity type
        
        Only include entities that are clearly mentioned in the text, with no speculations.
        Only return the JSON array with no additional text.
        
        Text to analyze:
        --------------
        {text}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    {"role": "system", "content": "You extract structured data from text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'(\[.*\])', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                entities = json.loads(json_str)
                
                # Normalize entity types
                for entity in entities:
                    if "type" in entity:
                        entity["type"] = entity["type"].capitalize()
                
                return entities
            else:
                print("Failed to extract JSON from LLM response")
                return []
                
        except Exception as e:
            print(f"Error in OpenAI entity extraction: {e}")
            return []
    
    def _rule_based_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using simple rule-based methods.
        Used as a fallback when LLM is not available.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entity dictionaries
        """
        entities = []
        
        # Extract people (simple pattern: capitalized names)
        name_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)'
        names = set(re.findall(name_pattern, text))
        
        for name in names:
            # Skip common false positives (e.g., months, days, etc.)
            if name in ["Monday Morning", "Tuesday Afternoon", "Wednesday Evening", 
                       "Thursday Night", "Friday Morning", "Saturday Afternoon", 
                       "Sunday Evening"]:
                continue
                
            entities.append({
                "id": str(uuid.uuid4()),
                "type": "Person",
                "name": name
            })
        
        # Extract organizations (simple pattern: consecutive capitalized words)
        org_pattern = r'([A-Z][a-zA-Z]+ (?:[A-Z][a-zA-Z]+ )*(?:Inc|LLC|Ltd|Corporation|Company|Organization|Foundation|Association))'
        orgs = set(re.findall(org_pattern, text))
        
        for org in orgs:
            entities.append({
                "id": str(uuid.uuid4()),
                "type": "Organization",
                "name": org
            })
        
        # Extract project names
        project_pattern = r'(Project [A-Z][a-zA-Z]*|[A-Z][a-zA-Z]+ Project)'
        projects = set(re.findall(project_pattern, text))
        
        for project in projects:
            entities.append({
                "id": str(uuid.uuid4()),
                "type": "Project",
                "name": project
            })
        
        # Extract concepts (simple approach: look for defined terms)
        concept_patterns = [
            r'"([^"]+)" (?:is|refers to|means)',
            r'([A-Z][a-zA-Z ]+) is (?:defined as|a concept)'
        ]
        
        concepts = set()
        for pattern in concept_patterns:
            concepts.update(re.findall(pattern, text))
        
        for concept in concepts:
            entities.append({
                "id": str(uuid.uuid4()),
                "type": "Concept",
                "name": concept
            })
        
        return entities
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        # Truncate text if too long
        max_length = 8000  # Arbitrary limit to avoid token limits
        if len(text) > max_length:
            text = text[:max_length]
        
        # Try Hugging Face model (Stella)
        if self.embedding_model == "huggingface_loaded" and self.hf_model is not None and self.tokenizer is not None:
            try:
                import torch
                # Process with Hugging Face model
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.hf_model(**inputs)
                
                # Use mean pooling as recommended for Stella
                # Get attention mask to handle padding properly
                attention_mask = inputs['attention_mask']
                # Get token embeddings from last hidden state
                token_embeddings = outputs.last_hidden_state
                
                # Apply attention mask to get a proper mean
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
                # Convert to list and return
                embeddings = mean_embeddings.squeeze().cpu().numpy()
                print(f"Generated embedding with Stella model, shape: {embeddings.shape}")
                return embeddings.tolist()
            except Exception as e:
                print(f"Error generating embeddings with Hugging Face model: {e}")
        
        # Try sentence-transformers if available
        if isinstance(self.embedding_model, object) and hasattr(self.embedding_model, 'encode'):
            try:
                embeddings = self.embedding_model.encode(text)
                print(f"Generated embedding with sentence-transformers, shape: {embeddings.shape}")
                return embeddings.tolist()
            except Exception as e:
                print(f"Error generating embeddings with sentence-transformers: {e}")
        
        # If we reach here, try to use OpenAI as fallback
        try:
            if "openai" in sys.modules:
                import openai
                openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                if openai_client is not None:
                    response = openai_client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small"
                    )
                    print(f"Generated embedding with OpenAI as fallback")
                    return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding as fallback: {e}")
        
        # As a last resort, try to load and use sentence-transformers if not already tried
        if not isinstance(self.embedding_model, object) and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                fallback_model = SentenceTransformer('all-mpnet-base-v2')
                embeddings = fallback_model.encode(text)
                print(f"Generated embedding with fallback sentence-transformers")
                return embeddings.tolist()
            except Exception as e:
                print(f"Error with fallback embedding generation: {e}")
        
        print("Warning: All embedding methods failed. Returning None.")
        return None
    
    def categorize_text(self, text: str) -> Dict[str, Any]:
        """
        Categorize text by topic, type, and other attributes.
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary with categorization information
        """
        # Default categorization
        categorization = {
            "type": "note",
            "topics": [],
            "sentiment": "neutral"
        }
        
        if not self.llm_enabled or not self.llm_client:
            # Use simple rule-based categorization
            return self._rule_based_categorization(text)
        
        try:
            # Use LLM for categorization
            if self.llm_provider == "anthropic":
                return self._categorize_with_anthropic(text)
            elif self.llm_provider == "openai":
                return self._categorize_with_openai(text)
            else:
                return self._rule_based_categorization(text)
        except Exception as e:
            print(f"Error in text categorization: {e}")
            return self._rule_based_categorization(text)
    
    def _categorize_with_anthropic(self, text: str) -> Dict[str, Any]:
        """
        Categorize text using Anthropic's Claude.
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary with categorization information
        """
        prompt = f"""
        Analyze the following text and categorize it based on:
        1. Type (note, meeting, journal, task, research, etc.)
        2. Topics (list of main topics discussed)
        3. Sentiment (positive, negative, neutral)
        
        Return the results as a JSON object with these properties:
        - type: The content type
        - topics: Array of main topics
        - sentiment: Overall sentiment
        
        Only return the JSON object with no additional text.
        
        Text to analyze:
        --------------
        {text}
        """
        
        try:
            response = self.llm_client.messages.create(
                model=self.config["llm"]["model"],
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                categorization = json.loads(json_str)
                return categorization
            else:
                print("Failed to extract JSON from LLM response")
                return self._rule_based_categorization(text)
                
        except Exception as e:
            print(f"Error in Anthropic categorization: {e}")
            return self._rule_based_categorization(text)
    
    def _categorize_with_openai(self, text: str) -> Dict[str, Any]:
        """
        Categorize text using OpenAI's API.
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary with categorization information
        """
        prompt = f"""
        Analyze the following text and categorize it based on:
        1. Type (note, meeting, journal, task, research, etc.)
        2. Topics (list of main topics discussed)
        3. Sentiment (positive, negative, neutral)
        
        Return the results as a JSON object with these properties:
        - type: The content type
        - topics: Array of main topics
        - sentiment: Overall sentiment
        
        Only return the JSON object with no additional text.
        
        Text to analyze:
        --------------
        {text}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[
                    {"role": "system", "content": "You analyze and categorize text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                categorization = json.loads(json_str)
                return categorization
            else:
                print("Failed to extract JSON from LLM response")
                return self._rule_based_categorization(text)
                
        except Exception as e:
            print(f"Error in OpenAI categorization: {e}")
            return self._rule_based_categorization(text)
    
    def _rule_based_categorization(self, text: str) -> Dict[str, Any]:
        """
        Categorize text using simple rule-based methods.
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary with categorization information
        """
        # Default values
        categorization = {
            "type": "note",
            "topics": [],
            "sentiment": "neutral"
        }
        
        # Content type detection
        if re.search(r'\bmeeting\b|\bdiscussion\b|\battendees\b', text, re.IGNORECASE):
            categorization["type"] = "meeting"
        elif re.search(r'\btask\b|\btodo\b|\baction item\b|\bcomplete by\b', text, re.IGNORECASE):
            categorization["type"] = "task"
        elif re.search(r'\bjournal\b|\bdear diary\b|\btoday i\b', text, re.IGNORECASE):
            categorization["type"] = "journal"
        elif re.search(r'\bresearch\b|\bstudy\b|\banalysis\b|\bfindings\b', text, re.IGNORECASE):
            categorization["type"] = "research"
        
        # Topic extraction (simple approach)
        # Find capitalized phrases that might be topics
        topic_candidates = re.findall(r'([A-Z][a-zA-Z ]{2,20})', text)
        topics = list(set(topic_candidates))[:5]  # Limit to 5 topics
        if topics:
            categorization["topics"] = topics
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'happy', 'best']
        negative_words = ['bad', 'poor', 'negative', 'failure', 'worst', 'problem', 'difficult']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if f" {word} " in f" {text_lower} ")
        negative_count = sum(1 for word in negative_words if f" {word} " in f" {text_lower} ")
        
        if positive_count > negative_count + 2:
            categorization["sentiment"] = "positive"
        elif negative_count > positive_count + 2:
            categorization["sentiment"] = "negative"
        
        return categorization