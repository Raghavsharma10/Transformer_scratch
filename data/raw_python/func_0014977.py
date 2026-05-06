def get_version():
    """Return version string."""
    with io.open('grammar_check/__init__.py', encoding='utf-8') as input_file:
        for line in input_file:
            if line.startswith('__version__'):
                return ast.parse(line).body[0].value.s