def list_build_configuration_sets(page_size=200, page_index=0, sort="", q=""):
    """
    List all build configuration sets
    """
    data = list_build_configuration_sets_raw(page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)