def update_license(license_id, **kwargs):
    """
    Replace the License with given ID with a new License
    """
    updated_license = pnc_api.licenses.get_specific(id=license_id).content

    for key, value in iteritems(kwargs):
        if value:
            setattr(updated_license, key, value)

    response = utils.checked_api_call(
        pnc_api.licenses,
        'update',
        id=int(license_id),
        body=updated_license)
    if response:
        return utils.format_json(response.content)