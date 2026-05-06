def add_dependency(id=None, name=None, dependency_id=None, dependency_name=None):
    """
    Add an existing BuildConfiguration as a dependency to another BuildConfiguration.
    """
    data = add_dependency_raw(id, name, dependency_id, dependency_name)
    if data:
        return utils.format_json_list(data)