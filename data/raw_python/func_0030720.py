def list_products(page_size=200, page_index=0, sort="", q=""):
    """
    List all Products
    """
    content = list_products_raw(page_size, page_index, sort, q)
    if content:
        return utils.format_json_list(content)