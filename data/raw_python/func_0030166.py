def list_build_set_records(id=None, name=None, page_size=200, page_index=0, sort="", q=""):
    """
    List all build set records for a BuildConfigurationSet
    """
    content = list_build_set_records_raw(id, name, page_size, page_index, sort, q)
    if content:
        return utils.format_json_list(content)