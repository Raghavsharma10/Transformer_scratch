def list_dependency_artifacts(id, page_size=200, page_index=0, sort="", q=""):
    """
    List dependency artifacts associated with a BuildRecord
    """
    data = list_dependency_artifacts_raw(id, page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)