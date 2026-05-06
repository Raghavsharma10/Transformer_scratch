def add_build_configuration_to_set(
        set_id=None, set_name=None, config_id=None, config_name=None):
    """
    Add a build configuration to an existing BuildConfigurationSet
    """
    content = add_build_configuration_to_set_raw(set_id, set_name, config_id, config_name)
    if content:
        return utils.format_json(content)