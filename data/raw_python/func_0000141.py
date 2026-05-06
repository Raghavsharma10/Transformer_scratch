def load(config_path: str):
    """
    Load a configuration and keep it alive for the given context

    :param config_path: path to a configuration file
    """
    # we bind the config to _ to keep it alive
    if os.path.splitext(config_path)[1] in ('.yaml', '.yml'):
        _ = load_yaml_configuration(config_path, translator=PipelineTranslator())
    elif os.path.splitext(config_path)[1] == '.py':
        _ = load_python_configuration(config_path)
    else:
        raise ValueError('Unknown configuration extension: %r' % os.path.splitext(config_path)[1])
    yield