def update_build_configuration(id, **kwargs):
    """
    Update an existing BuildConfiguration with new information

    :param id: ID of BuildConfiguration to update
    :param name: Name of BuildConfiguration to update
    :return:
    """
    data = update_build_configuration_raw(id, **kwargs)
    if data:
        return utils.format_json(data)