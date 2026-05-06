def list_build_configurations_for_product_version(product_id, version_id, page_size=200, page_index=0, sort="", q=""):
    """
    List all BuildConfigurations associated with the given ProductVersion
    """
    data = list_build_configurations_for_project_raw(product_id, version_id, page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)