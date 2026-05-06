def create_license(**kwargs):
    """
    Create a new License
    """
    License = create_license_object(**kwargs)
    response = utils.checked_api_call(pnc_api.licenses, 'create_new', body=License)
    if response:
        return utils.format_json(response.content)