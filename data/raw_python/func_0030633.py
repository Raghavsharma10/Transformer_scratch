def update_release(id, **kwargs):
    """
    Update an existing ProductRelease with new information
    """
    data = update_release_raw(id, **kwargs)
    if data:
        return utils.format_json(data)