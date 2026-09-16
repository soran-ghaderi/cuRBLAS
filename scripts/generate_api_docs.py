#!/usr/bin/env python3
"""
turns doxygen xml output into markdown pages mkdocs can serve.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml


def parse_doxygen_xml(xml_path):
    """read the doxygen xml dump into plain dicts"""
    docs = {}

    index_file = xml_path / "index.xml"
    if not index_file.exists():
        print(f"Error: {index_file} not found. Make sure Doxygen has been run.")
        return docs

    tree = ET.parse(index_file)
    root = tree.getroot()

    for compound in root.findall('.//compound'):
        kind = compound.get('kind')
        if kind in ['class', 'struct', 'namespace', 'file']:
            refid = compound.get('refid')
            name = compound.find('name').text

            xml_file = xml_path / f"{refid}.xml"
            if xml_file.exists():
                docs[name] = parse_compound_xml(xml_file, kind)

    return docs


def parse_compound_xml(xml_file, kind):
    """pull one compound (class, struct, namespace) out of its xml file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    doc_info = {
        'kind': kind,
        'name': '',
        'brief': '',
        'detailed': '',
        'members': []
    }

    compound_def = root.find('.//compounddef')
    if compound_def is not None:
        doc_info['name'] = compound_def.find('compoundname').text if compound_def.find('compoundname') is not None else ''

        brief = compound_def.find('briefdescription')
        if brief is not None:
            doc_info['brief'] = extract_text(brief)

        detailed = compound_def.find('detaileddescription')
        if detailed is not None:
            doc_info['detailed'] = extract_text(detailed)

        for member in compound_def.findall('.//memberdef'):
            member_info = parse_member(member)
            if member_info:
                doc_info['members'].append(member_info)

    return doc_info


def parse_member(member_element):
    """pull one member (function, variable, enum) out of the xml"""
    member_info = {
        'kind': member_element.get('kind', ''),
        'name': '',
        'type': '',
        'brief': '',
        'detailed': '',
        'params': []
    }

    name_elem = member_element.find('name')
    if name_elem is not None:
        member_info['name'] = name_elem.text

    type_elem = member_element.find('type')
    if type_elem is not None:
        member_info['type'] = extract_text(type_elem)

    brief = member_element.find('briefdescription')
    if brief is not None:
        member_info['brief'] = extract_text(brief)

    detailed = member_element.find('detaileddescription')
    if detailed is not None:
        member_info['detailed'] = extract_text(detailed)

    for param in member_element.findall('.//param'):
        param_info = {
            'type': '',
            'name': '',
            'description': ''
        }

        param_type = param.find('type')
        if param_type is not None:
            param_info['type'] = extract_text(param_type)

        param_name = param.find('declname')
        if param_name is not None:
            param_info['name'] = param_name.text

        member_info['params'].append(param_info)

    return member_info


def extract_text(element):
    """flatten an element's text, nested children included"""
    if element is None:
        return ""

    text = element.text or ""
    for child in element:
        text += extract_text(child)
        if child.tail:
            text += child.tail

    return text.strip()


def generate_markdown_docs(docs, output_dir):
    """write out every markdown page"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generate_overview_page(docs, output_path)

    generated_files = []
    for name, doc_info in docs.items():
        if doc_info['kind'] in ['class', 'struct', 'namespace']:
            filename = generate_compound_page(name, doc_info, output_path)
            if filename:
                generated_files.append((name, filename, doc_info))

    return generated_files


def generate_overview_page(docs, output_path):
    """write the api index page"""
    content = ["# API Reference", "", "## Overview", ""]

    classes = []
    structs = []
    namespaces = []
    files = []

    for name, doc_info in docs.items():
        if doc_info['kind'] == 'class':
            classes.append((name, doc_info))
        elif doc_info['kind'] == 'struct':
            structs.append((name, doc_info))
        elif doc_info['kind'] == 'namespace':
            namespaces.append((name, doc_info))
        elif doc_info['kind'] == 'file':
            files.append((name, doc_info))

    if namespaces:
        content.extend(["## Namespaces", ""])
        for name, doc_info in sorted(namespaces):
            content.append(f"- [{name}]({name.replace('::', '_').lower()}.md) - {doc_info['brief']}")
        content.append("")

    if classes:
        content.extend(["## Classes", ""])
        for name, doc_info in sorted(classes):
            content.append(f"- [{name}]({name.replace('::', '_').lower()}.md) - {doc_info['brief']}")
        content.append("")

    if structs:
        content.extend(["## Structures", ""])
        for name, doc_info in sorted(structs):
            content.append(f"- [{name}]({name.replace('::', '_').lower()}.md) - {doc_info['brief']}")
        content.append("")

    with open(output_path / "index.md", "w") as f:
        f.write("\n".join(content))


def generate_compound_page(name, doc_info, output_path):
    """write one page for a class, struct or namespace"""
    filename = name.replace('::', '_').lower() + '.md'

    content = [f"# {name}", ""]

    if doc_info['brief']:
        content.extend([doc_info['brief'], ""])

    if doc_info['detailed']:
        content.extend(["## Description", "", doc_info['detailed'], ""])

    functions = []
    variables = []

    for member in doc_info['members']:
        if member['kind'] in ['function', 'define']:
            functions.append(member)
        elif member['kind'] in ['variable', 'enum']:
            variables.append(member)

    if functions:
        content.extend(["## Functions", ""])
        for func in functions:
            content.extend(generate_function_doc(func))
        content.append("")

    if variables:
        content.extend(["## Variables", ""])
        for var in variables:
            content.extend(generate_variable_doc(var))
        content.append("")

    with open(output_path / filename, "w") as f:
        f.write("\n".join(content))

    return filename


def generate_function_doc(func):
    """markdown block for one function"""
    content = [f"### {func['name']}", ""]

    params_str = ", ".join([f"{p['type']} {p['name']}" for p in func['params']])
    content.append(f"```c")
    content.append(f"{func['type']} {func['name']}({params_str})")
    content.append("```")
    content.append("")

    if func['brief']:
        content.extend([func['brief'], ""])

    if func['detailed']:
        content.extend(["**Description:**", "", func['detailed'], ""])

    if func['params']:
        content.extend(["**Parameters:**", ""])
        for param in func['params']:
            content.append(f"- `{param['name']}` ({param['type']}): {param['description']}")
        content.append("")

    return content


def generate_variable_doc(var):
    """markdown block for one variable"""
    content = [f"### {var['name']}", ""]

    content.append(f"```c")
    content.append(f"{var['type']} {var['name']}")
    content.append("```")
    content.append("")

    if var['brief']:
        content.extend([var['brief'], ""])

    if var['detailed']:
        content.extend(["**Description:**", "", var['detailed'], ""])

    return content


def update_mkdocs_navigation(generated_files, mkdocs_path):
    """point the api reference nav at the pages we just wrote"""
    try:
        with open(mkdocs_path, 'r') as f:
            mkdocs_config = yaml.safe_load(f)

        api_nav = []
        api_nav.append({'Overview': 'api/index.md'})

        classes = []
        structs = []
        namespaces = []

        for name, filename, doc_info in generated_files:
            nav_item = {name: f'api/{filename}'}

            if doc_info['kind'] == 'class':
                classes.append(nav_item)
            elif doc_info['kind'] == 'struct':
                structs.append(nav_item)
            elif doc_info['kind'] == 'namespace':
                namespaces.append(nav_item)

        if namespaces:
            if api_nav:
                api_nav.extend(sorted(namespaces, key=lambda x: list(x.keys())[0]))

        if classes:
            api_nav.extend(sorted(classes, key=lambda x: list(x.keys())[0]))

        if structs:
            api_nav.extend(sorted(structs, key=lambda x: list(x.keys())[0]))

        nav = mkdocs_config.get('nav', [])

        for i, section in enumerate(nav):
            if isinstance(section, dict) and 'API Reference' in section:
                nav[i] = {'API Reference': api_nav}
                break
        else:
            insert_pos = len(nav)
            for i, section in enumerate(nav):
                if isinstance(section, dict) and 'User Guide' in section:
                    insert_pos = i + 1
                    break
            nav.insert(insert_pos, {'API Reference': api_nav})

        mkdocs_config['nav'] = nav

        with open(mkdocs_path, 'w') as f:
            yaml.dump(mkdocs_config, f, default_flow_style=False, sort_keys=False, indent=2)

        print(f"Updated mkdocs.yml navigation with {len(generated_files)} API pages")

    except Exception as e:
        print(f"Warning: Could not update mkdocs.yml navigation: {e}")
        print("You may need to manually update the API Reference section")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python generate_api_docs.py <doxygen_xml_dir> <output_dir> [mkdocs_yml_path]")
        sys.exit(1)

    xml_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    mkdocs_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("mkdocs.yml")

    if not xml_dir.exists():
        print(f"Error: XML directory {xml_dir} does not exist.")
        sys.exit(1)

    print(f"Parsing Doxygen XML from {xml_dir}...")
    docs = parse_doxygen_xml(xml_dir)

    print(f"Generating Markdown documentation to {output_dir}...")
    generated_files = generate_markdown_docs(docs, output_dir)

    if mkdocs_path.exists():
        print(f"Updating navigation in {mkdocs_path}...")
        update_mkdocs_navigation(generated_files, mkdocs_path)
    else:
        print(f"Warning: {mkdocs_path} not found. Navigation not updated.")

    print("API documentation generation complete!")


if __name__ == "__main__":
    main()
