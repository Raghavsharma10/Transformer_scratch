def delete_license(license_id):
    """
    Delete a License by ID
    """

    response = utils.checked_api_call(pnc_api.licenses, 'delete', id=license_id)
    if response:
        return utils.format_json(response.content)