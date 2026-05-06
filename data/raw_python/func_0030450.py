def print_yaml(o):
    """Pretty print an object as YAML."""
    print(yaml.dump(o, default_flow_style=False, indent=4, encoding='utf-8'))