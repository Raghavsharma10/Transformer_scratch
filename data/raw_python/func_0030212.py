def list_records_for_build_configuration(id=None, name=None, page_size=200, page_index=0, sort="", q=""):
    """
    List all BuildRecords for a given BuildConfiguration
    """
    data = list_records_for_build_configuration_raw(id, name, page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)