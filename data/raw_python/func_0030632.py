def list_product_releases(page_size=200, page_index=0, sort="", q=""):
    """
    List all ProductReleases
    """
    data = list_product_releases_raw(page_size, page_index, sort, q)
    if data:
        return utils.format_json_list(data)