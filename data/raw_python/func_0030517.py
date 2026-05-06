def get_build_configuration(id=None, name=None):
    """
    Retrieve a specific BuildConfiguration
    """
    data = get_build_configuration_raw(id, name)
    if data:
        return utils.format_json(data)