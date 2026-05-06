def update_product_version(id, **kwargs):
    """
    Update the ProductVersion with ID id with new values.
    """
    content = update_product_version_raw(id, **kwargs)
    if content:
        return utils.format_json(content)