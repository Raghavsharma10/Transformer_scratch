def create_product_version(product_id, version, **kwargs):
    """
    Create a new ProductVersion.
    Each ProductVersion represents a supported product release stream, which includes milestones and releases typically associated with a single major.minor version of a Product.
    Follows the Red Hat product support cycle, and typically includes Alpha, Beta, GA, and CP releases with the same major.minor version.

    Example:
    ProductVersion 1.0 includes the following releases:
    1.0.Beta1, 1.0.GA, 1.0.1, etc.
    """
    data = create_product_version_raw(product_id, version, **kwargs)
    if data:
        return utils.format_json(data)