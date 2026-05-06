def get_product(id=None, name=None):
    """
    Get a specific Product by name or ID
    """
    content = get_product_raw(id, name)
    if content:
        return utils.format_json(content)