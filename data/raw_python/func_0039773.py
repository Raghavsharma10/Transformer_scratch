def delete_script(delete=None):  # noqa: E501
    """Delete a script

    Delete a script # noqa: E501

    :param delete: The data needed to delete this script
    :type delete: dict | bytes

    :rtype: Response
    """
    if connexion.request.is_json:
        delete = Delete.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'