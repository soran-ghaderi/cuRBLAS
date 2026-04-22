#!/usr/bin/env python3
"""
Script to automatically update mkdocs.yml navigation with generated API documentation.
This script can be used standalone or as a fallback when generate_api_docs.py doesn't run.
"""

import os
import yaml
from pathlib import Path


def analyze_markdown_file(file_path):
    """Analyze a markdown file to determine its type and extract title."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract title from first heading
        lines = content.split('\n')
        title = None
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        if not title:
            # Fallback to filename
            title = file_path.stem.replace('_', ' ').title()
        
        # Determine type based on content patterns
        content_lower = content.lower()
        if 'namespace' in content_lower or '::' in content:
            return title, 'namespace'
        elif 'class' in content_lower or 'struct' in content_lower:
            # Try to distinguish between class and struct
            if 'struct' in content_lower and content_lower.count('struct') > content_lower.count('class'):
                return title, 'struct'
            else:
                return title, 'class'
        else:
            return title, 'other'
            
    except Exception:
        # Fallback
        title = file_path.stem.replace('_', ' ').title()
        return title, 'other'


def update_api_navigation():
    """Update the API Reference section in mkdocs.yml with generated files."""
    
    # Find mkdocs.yml
    mkdocs_file = Path("mkdocs.yml")
    if not mkdocs_file.exists():
        print("Error: mkdocs.yml not found")
        return
    
    # Read mkdocs.yml
    try:
        with open(mkdocs_file, 'r') as f:
            mkdocs_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading mkdocs.yml: {e}")
        return
    
    # Find API documentation files
    api_dir = Path("docs/api")
    if not api_dir.exists():
        print("Error: docs/api directory not found")
        return
    
    # Analyze API files
    namespaces = []
    classes = []
    structs = []
    others = []
    
    for md_file in api_dir.glob("*.md"):
        if md_file.name != "index.md":
            title, file_type = analyze_markdown_file(md_file)
            nav_item = {title: f"api/{md_file.name}"}
            
            if file_type == 'namespace':
                namespaces.append(nav_item)
            elif file_type == 'class':
                classes.append(nav_item)
            elif file_type == 'struct':
                structs.append(nav_item)
            else:
                others.append(nav_item)
    
    # Build API Reference navigation structure
    api_nav = []
    api_nav.append({'Overview': 'api/index.md'})
    
    # Add organized sections (sorted alphabetically)
    if namespaces:
        api_nav.extend(sorted(namespaces, key=lambda x: list(x.keys())[0].lower()))
    
    if classes:
        api_nav.extend(sorted(classes, key=lambda x: list(x.keys())[0].lower()))
    
    if structs:
        api_nav.extend(sorted(structs, key=lambda x: list(x.keys())[0].lower()))
    
    if others:
        api_nav.extend(sorted(others, key=lambda x: list(x.keys())[0].lower()))
    
    # Update navigation in mkdocs config
    nav = mkdocs_config.get('nav', [])
    
    # Find and update the API Reference section
    updated = False
    for i, section in enumerate(nav):
        if isinstance(section, dict) and 'API Reference' in section:
            nav[i] = {'API Reference': api_nav}
            updated = True
            break
    
    if not updated:
        # If API Reference section doesn't exist, add it
        # Find a good position (after User Guide if it exists)
        insert_pos = len(nav)
        for i, section in enumerate(nav):
            if isinstance(section, dict) and 'User Guide' in section:
                insert_pos = i + 1
                break
        nav.insert(insert_pos, {'API Reference': api_nav})
    
    mkdocs_config['nav'] = nav
    
    # Write updated mkdocs.yml
    try:
        with open(mkdocs_file, 'w') as f:
            yaml.dump(mkdocs_config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        total_files = len(namespaces) + len(classes) + len(structs) + len(others)
        print(f"Updated API navigation with {total_files} files:")
        print(f"  - {len(namespaces)} namespaces")
        print(f"  - {len(classes)} classes") 
        print(f"  - {len(structs)} structs")
        print(f"  - {len(others)} other files")
        
    except Exception as e:
        print(f"Error writing mkdocs.yml: {e}")


if __name__ == "__main__":
    update_api_navigation() 