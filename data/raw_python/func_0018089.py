def initialize():
    """
    Initialize the configuration system by installing YAML handlers.
    Automatically done on first call to load() specified in this file.
    """
    global is_initialized
    # Add the custom multi-constructor
    yaml.add_multi_constructor('!obj:', multi_constructor)
    yaml.add_multi_constructor('!pkl:', multi_constructor_pkl)
    yaml.add_multi_constructor('!import:', multi_constructor_import)
    yaml.add_multi_constructor('!include:', multi_constructor_include)

    def import_constructor(loader, node):
        value = loader.construct_scalar(node)
        return try_to_import(value)

    yaml.add_constructor('!import', import_constructor)
    yaml.add_implicit_resolver(
        '!import',
        re.compile(r'(?:[a-zA-Z_][\w_]+\.)+[a-zA-Z_][\w_]+')
    )
    is_initialized = True