def get_license(id):
    """
    Get a specific License by either ID or fullname
    """
    response = utils.checked_api_call(
        pnc_api.licenses, 'get_specific', id= id)
    if response:
        return utils.format_json(response.content)