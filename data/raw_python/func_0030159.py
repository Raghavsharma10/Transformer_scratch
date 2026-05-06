def get_build_configuration_set(id=None, name=None):
    """
    Get a specific BuildConfigurationSet by name or ID
    """
    content = get_build_configuration_set_raw(id, name)
    if content:
        return utils.format_json(content)