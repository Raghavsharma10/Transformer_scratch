def list_build_configurations_for_product(id=None, name=None, page_size=200, page_index=0, sort="", q=""):
    """
    List all BuildConfigurations associated with the given Product.
    """
    data = list_build_configurations_for_product_raw(id, name, page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)