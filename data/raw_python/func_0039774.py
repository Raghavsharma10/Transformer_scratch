def rename_script(rename=None):  # noqa: E501
    """Rename a script

    Rename a script # noqa: E501

    :param rename: The data needed to save this script
    :type rename: dict | bytes

    :rtype: Response
    """
    if connexion.request.is_json:
        rename = Rename.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'