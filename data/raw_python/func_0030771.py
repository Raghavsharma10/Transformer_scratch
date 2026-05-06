def get_project(id=None, name=None):
    """
    Get a specific Project by ID or name
    """
    content = get_project_raw(id, name)
    if content:
        return utils.format_json(content)