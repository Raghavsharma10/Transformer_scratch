def add_product_version_to_build_configuration(id=None, name=None, product_version_id=None):
    """
    Associate an existing ProductVersion with a BuildConfiguration
    """
    data = remove_product_version_from_build_configuration_raw(id, name, product_version_id)
    if data:
        return utils.format_json_list(data)