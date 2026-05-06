def list_build_configurations(page_size=200, page_index=0, sort="", q=""):
    """
    List all BuildConfigurations
    """
    data = list_build_configurations_raw(page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)