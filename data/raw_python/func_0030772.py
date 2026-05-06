def delete_project(id=None, name=None):
    """
    Delete a Project by ID or name.
    """
    content = delete_project_raw(id, name)
    if content:
        return utils.format_json(content)