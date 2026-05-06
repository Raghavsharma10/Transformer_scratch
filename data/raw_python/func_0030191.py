def list_records_for_build_config_set(id, page_size=200, page_index=0, sort="", q=""):
    """
    Get a list of BuildRecords for the given BuildConfigSetRecord
    """
    data = list_records_for_build_config_set_raw(id, page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)