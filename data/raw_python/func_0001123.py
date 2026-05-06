def _product_filter(products) -> str:
    """Calculate the product filter."""
    _filter = 0
    for product in {PRODUCTS[p] for p in products}:
        _filter += product
    return format(_filter, "b")[::-1]