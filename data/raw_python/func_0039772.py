def create_script(create=None):  # noqa: E501
    """Create a new script

    Create a new script # noqa: E501

    :param create: The data needed to create this script
    :type create: dict | bytes

    :rtype: Response
    """
    if connexion.request.is_json:
        create = Create.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'