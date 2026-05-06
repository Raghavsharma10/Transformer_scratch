def update_product(product_id, **kwargs):
    """
    Update a Product with new information
    """
    content = update_product_raw(product_id, **kwargs)
    if content:
        return utils.format_json(content)