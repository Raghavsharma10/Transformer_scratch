def list_build_configurations_for_set(id=None, name=None, page_size=200, page_index=0, sort="", q=""):
    """
    List all build configurations in a given BuildConfigurationSet.
    """
    content = list_build_configurations_for_set_raw(id, name, page_size, page_index, sort, q)
    if content:
        return utils.format_json_list(content)