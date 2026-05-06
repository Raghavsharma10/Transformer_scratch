def userToJson(user):
    """Returns a serializable User dict

    :param user: User to get info for
    :type user: User
    :returns: dict
    """
    obj = {
        'id': user.id,
        'username': user.username,
        'name': user.get_full_name(),
        'email': user.email,
    }

    return obj