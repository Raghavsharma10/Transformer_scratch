def list_build_configuration_set_records(page_size=200, page_index=0, sort="", q=""):
    """
    List all build configuration set records.
    """
    data = list_build_configuration_set_records_raw(page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)