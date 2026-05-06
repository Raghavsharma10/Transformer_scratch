def delete_build_configuration(id=None, name=None):
    """
    Delete an existing BuildConfiguration
    :param id:
    :param name:
    :return:
    """
    data = delete_build_configuration_raw(id, name)
    if data:
        return utils.format_json(data)