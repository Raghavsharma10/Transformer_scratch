def get_environment(id=None, name=None):
    """
    Get a specific Environment by name or ID
    """
    data = get_environment_raw(id, name)
    if data:
        return utils.format_json(data)