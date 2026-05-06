def create_product(name, abbreviation, **kwargs):
    """
    Create a new Product
    """
    data = create_product_raw(name, abbreviation, **kwargs)
    if data:
        return utils.format_json(data)