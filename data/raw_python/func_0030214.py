def list_built_artifacts(id, page_size=200, page_index=0, sort="", q=""):
    """
    List Artifacts associated with a BuildRecord
    """
    data = list_built_artifacts_raw(id, page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)