"""
Local Storage Module
===================

Handles local file storage for Knowledge Commons, including:
- Saving and loading markdown files with YAML frontmatter
- Managing JSON-LD documents
- Organizing content based on metadata
"""

import os
import json
import shutil
import frontmatter
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import yaml


class LocalStorage:
    """Manages local storage of knowledge commons content."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local storage manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.content_path = Path(config["storage"]["local"]["content_path"])
        self.content_path.mkdir(exist_ok=True, parents=True)
        
        # Create standard directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create standard content directories if they don't exist."""
        standard_dirs = ["notes", "people", "organizations", "projects", "events", "tasks"]
        for directory in standard_dirs:
            (self.content_path / directory).mkdir(exist_ok=True)
    
    def save_markdown(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save content as a markdown file with YAML frontmatter.
        
        Args:
            content: The markdown content
            metadata: Metadata to store in YAML frontmatter
            
        Returns:
            Dict with information about the saved file
        """
        # Generate a filename if not provided
        filename = metadata.get("filename")
        if not filename:
            title = metadata.get("title", "Untitled Note")
            date_str = datetime.now().strftime("%Y%m%d")
            # Create a filename-safe version of the title
            safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title)
            safe_title = safe_title.replace(' ', '-').lower()
            filename = f"{date_str}-{safe_title}.md"
        
        # Determine target directory based on content type
        content_type = metadata.get("type", "note")
        if content_type == "note":
            target_dir = self.content_path / "notes"
        elif content_type == "person":
            target_dir = self.content_path / "people"
        elif content_type == "organization":
            target_dir = self.content_path / "organizations"
        elif content_type == "project":
            target_dir = self.content_path / "projects"
        elif content_type == "event":
            target_dir = self.content_path / "events"
        elif content_type == "task":
            target_dir = self.content_path / "tasks"
        else:
            target_dir = self.content_path / "misc"
        
        target_dir.mkdir(exist_ok=True)
        file_path = target_dir / filename
        
        # Ensure we don't overwrite existing files unless specified
        if file_path.exists() and not metadata.get("overwrite", False):
            base, ext = os.path.splitext(filename)
            counter = 1
            while file_path.exists():
                new_filename = f"{base}-{counter}{ext}"
                file_path = target_dir / new_filename
                counter += 1
        
        # Create the post with frontmatter and content
        post = frontmatter.Post(content, **metadata)
        
        # Write to file
        with open(file_path, 'wb') as f:
            frontmatter.dump(post, f)
        
        return {
            "path": file_path,
            "filename": file_path.name,
            "content_type": content_type
        }
    
    def load_markdown(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a markdown file with YAML frontmatter.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dict containing content and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the file with frontmatter
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        # Return a dictionary with content and metadata
        result = {
            "content": post.content,
            "metadata": dict(post.metadata),
            "path": file_path
        }
        
        return result
    
    def save_jsonld(self, filename: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save data as a JSON-LD file.
        
        Args:
            filename: Filename or path relative to content directory
            data: JSON-LD data to save
            
        Returns:
            Dict with information about the saved file
        """
        # Make sure it has the correct extension
        if not filename.endswith('.jsonld'):
            filename += '.jsonld'
        
        # Determine the full path
        if '/' in filename:
            # Contains a subdirectory, use as is
            file_path = self.content_path / filename
            # Ensure the directory exists
            file_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            # No subdirectory, place in misc
            file_path = self.content_path / "misc" / filename
            (self.content_path / "misc").mkdir(exist_ok=True)
        
        # Save the JSON-LD file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return {
            "path": file_path,
            "filename": file_path.name,
            "content_type": "jsonld"
        }
    
    def load_jsonld(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a JSON-LD file.
        
        Args:
            file_path: Path to the JSON-LD file
            
        Returns:
            Dict containing the JSON-LD data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "data": data,
            "path": file_path
        }
    
    def save_rdf(self, filename: str, data: str, format: str = "turtle") -> Dict[str, Any]:
        """
        Save RDF data to a file.
        
        Args:
            filename: Filename or path relative to content directory
            data: RDF data as string
            format: RDF format (turtle, xml, n3, etc.)
            
        Returns:
            Dict with information about the saved file
        """
        # Determine file extension based on format
        if format == "turtle":
            ext = ".ttl"
        elif format == "xml":
            ext = ".rdf"
        elif format == "n3":
            ext = ".n3"
        elif format == "ntriples":
            ext = ".nt"
        elif format == "jsonld":
            ext = ".jsonld"
        else:
            ext = f".{format}"
        
        # Make sure it has the correct extension
        if not filename.endswith(ext):
            filename += ext
        
        # Determine the full path
        if '/' in filename:
            # Contains a subdirectory, use as is
            file_path = self.content_path / filename
            # Ensure the directory exists
            file_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            # No subdirectory, place in misc
            file_path = self.content_path / "misc" / filename
            (self.content_path / "misc").mkdir(exist_ok=True)
        
        # Save the RDF file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
        
        return {
            "path": file_path,
            "filename": file_path.name,
            "content_type": "rdf",
            "format": format
        }
    
    def load_rdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an RDF file.
        
        Args:
            file_path: Path to the RDF file
            
        Returns:
            Dict containing the RDF data as string
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        # Determine format from extension
        suffix = file_path.suffix.lower()
        if suffix == '.ttl':
            format = "turtle"
        elif suffix == '.rdf':
            format = "xml"
        elif suffix == '.n3':
            format = "n3"
        elif suffix == '.nt':
            format = "ntriples"
        elif suffix == '.jsonld':
            format = "jsonld"
        else:
            format = suffix[1:]  # Remove the leading dot
        
        return {
            "data": data,
            "path": file_path,
            "format": format
        }
    
    def find_files(self, 
                   content_type: Optional[str] = None, 
                   tags: Optional[List[str]] = None,
                   search_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find files based on criteria.
        
        Args:
            content_type: Optional type of content to filter by
            tags: Optional list of tags to filter by
            search_text: Optional text to search for in content
            
        Returns:
            List of dictionaries with file information
        """
        results = []
        
        # Determine which directories to search
        if content_type:
            if content_type == "note":
                search_dirs = [self.content_path / "notes"]
            elif content_type == "person":
                search_dirs = [self.content_path / "people"]
            elif content_type == "organization":
                search_dirs = [self.content_path / "organizations"]
            elif content_type == "project":
                search_dirs = [self.content_path / "projects"]
            elif content_type == "event":
                search_dirs = [self.content_path / "events"]
            elif content_type == "task":
                search_dirs = [self.content_path / "tasks"]
            else:
                search_dirs = [self.content_path / "misc"]
        else:
            # Search all directories
            search_dirs = [self.content_path / dir_name for dir_name in 
                          ["notes", "people", "organizations", "projects", "events", "tasks", "misc"]]
        
        # Search for files
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for file_path in search_dir.glob("**/*"):
                if not file_path.is_file():
                    continue
                
                # Check file extension
                if file_path.suffix.lower() == ".md":
                    try:
                        file_data = self.load_markdown(file_path)
                        
                        # Filter by tags if specified
                        if tags:
                            file_tags = file_data["metadata"].get("tags", [])
                            if not file_tags:
                                continue
                                
                            # Convert to list if it's a string
                            if isinstance(file_tags, str):
                                file_tags = [tag.strip() for tag in file_tags.split(",")]
                                
                            # Check if any of the requested tags are in the file's tags
                            if not any(tag in file_tags for tag in tags):
                                continue
                        
                        # Search text if specified
                        if search_text:
                            if search_text.lower() not in file_data["content"].lower():
                                if not any(search_text.lower() in str(v).lower() 
                                        for v in file_data["metadata"].values()):
                                    continue
                        
                        # Add to results
                        results.append({
                            "path": file_path,
                            "filename": file_path.name,
                            "metadata": file_data["metadata"],
                            "content_type": file_data["metadata"].get("type", "note")
                        })
                    except Exception as e:
                        # Skip files that can't be parsed
                        print(f"Error parsing {file_path}: {e}")
                        continue
                
                elif file_path.suffix.lower() == ".jsonld":
                    # For JSON-LD files, just return basic info
                    results.append({
                        "path": file_path,
                        "filename": file_path.name,
                        "content_type": "jsonld"
                    })
        
        return results
    
    def count_files_by_type(self, file_type: str) -> int:
        """
        Count files of a specific type.
        
        Args:
            file_type: Type of file to count (markdown, jsonld, rdf)
            
        Returns:
            Number of files
        """
        count = 0
        
        if file_type == "markdown":
            extension = ".md"
        elif file_type == "jsonld":
            extension = ".jsonld"
        elif file_type == "rdf":
            extension = ".ttl"  # Count Turtle files as default RDF
        else:
            extension = f".{file_type}"
        
        # Recursively count files with the extension
        for root, dirs, files in os.walk(self.content_path):
            count += sum(1 for file in files if file.endswith(extension))
        
        return count
    
    def backup(self, backup_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Create a backup of the content directory.
        
        Args:
            backup_dir: Optional backup directory, uses config default if not specified
            
        Returns:
            Dict with information about the backup
        """
        if backup_dir is None:
            backup_dir = self.config["storage"]["local"].get("backup", {}).get("path")
            if backup_dir is None:
                backup_dir = Path(self.content_path.parent) / "backups"
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamped backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"content_backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Copy all files
        for item in self.content_path.glob("**/*"):
            if item.is_file():
                # Get the relative path
                rel_path = item.relative_to(self.content_path)
                # Create target directory if it doesn't exist
                target_dir = backup_path / rel_path.parent
                target_dir.mkdir(exist_ok=True, parents=True)
                # Copy the file
                shutil.copy2(item, target_dir / item.name)
        
        return {
            "backup_path": backup_path,
            "timestamp": timestamp,
            "file_count": sum(1 for _ in backup_path.glob("**/*") if _.is_file())
        }