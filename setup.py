#!/usr/bin/env python
"""
Knowledge Commons
================

A federated personal knowledge management system.
"""

import os
from setuptools import setup, find_packages

# Read version from __init__.py
version = {}
with open(os.path.join("knowledge_commons", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Main setup configuration
setup(
    name="knowledge-commons",
    version=version.get("__version__", "0.1.0"),
    description="A federated personal knowledge management system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/knowledge-commons",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "rdflib>=6.0.0",
        "rdflib-jsonld>=0.6.2",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "pydantic>=2.0.0",
        "requests>=2.28.0",
        "markdown>=3.4.0",
        "beautifulsoup4>=4.12.0",
        "python-frontmatter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "llm": [
            "anthropic>=0.8.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
        ],
        "vector": [
            "qdrant-client>=1.6.0",
            "chromadb>=0.4.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.23.0",
        ],
        "complete": [
            "anthropic>=0.8.0",
            "openai>=1.0.0",
            "sentence-transformers>=2.2.0",
            "qdrant-client>=1.6.0",
            "chromadb>=0.4.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.23.0",
            "langchain>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "knowledge-commons=knowledge_commons.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Office/Business :: Groupware",
        "Topic :: Text Processing :: Markup",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="knowledge-graph, rag, vector-database, federation, personal-knowledge-management, rdf, semantics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/knowledge-commons/issues",
        "Source": "https://github.com/yourusername/knowledge-commons",
        "Documentation": "https://github.com/yourusername/knowledge-commons/blob/main/README.md",
    },
    package_data={
        "knowledge_commons": ["py.typed", "*.yaml"],
    },
)