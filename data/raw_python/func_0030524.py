def remove_dependency(id=None, name=None, dependency_id=None, dependency_name=None):
    """
    Remove a BuildConfiguration from the dependency list of another BuildConfiguration
    """
    data = remove_dependency_raw(id, name, dependency_id, dependency_name)
    if data:
        return utils.format_json_list(data)