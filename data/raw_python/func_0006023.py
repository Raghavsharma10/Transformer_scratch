def load_config_vars(target_config, source_config):
    """Loads all attributes from source config into target config

    @type target_config: TestRunConfigManager
    @param target_config: Config to dump variables into
    @type source_config: TestRunConfigManager
    @param source_config: The other config
    @return: True
    """
    # Overwrite all attributes in config with new config
    for attr in dir(source_config):
        # skip all private class attrs
        if attr.startswith('_'):
            continue
        val = getattr(source_config, attr)
        if val is not None:
            setattr(target_config, attr, val)