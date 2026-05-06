def update_project(id, **kwargs):
    """
    Update an existing Project with new information
    """
    content = update_project_raw(id, **kwargs)
    if content:
        return utils.format_json(content)