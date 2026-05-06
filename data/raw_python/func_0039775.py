def save_script(save=None):  # noqa: E501
    """Save a script

    Save a script # noqa: E501

    :param save: The data needed to save this script
    :type save: dict | bytes

    :rtype: Response
    """
    if connexion.request.is_json:
        save = Save.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'